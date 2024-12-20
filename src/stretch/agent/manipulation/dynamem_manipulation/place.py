# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import math
import os
from typing import Tuple

import numpy as np
import scipy
from PIL import Image

from .image_publisher import DynamemCamera

Bbox = Tuple[int, int, int, int]


class Placing:
    def __init__(self, robot, detection_model, save_dir=None):
        self.camera = DynamemCamera(robot)
        self.detection_model = detection_model
        self.fx = self.camera.fx
        self.fy = self.camera.fy
        self.cx = self.camera.cx
        self.cy = self.camera.cy
        self.max_depth = 1.5
        self.tries = 1
        if save_dir is None:
            save_dir = "test"
        self.save_dir = save_dir

    def setup(self, text, head_tilt=-1):
        image, depth, _ = self.camera.capture_image()
        self.image = Image.fromarray(np.rot90(image, k=-1))
        self.depth = np.rot90(depth, k=-1)
        self.head_tilt = head_tilt
        self.query = text

    def get_3d_points(self):

        xmap, ymap = np.arange(self.depth.shape[1]), np.arange(self.depth.shape[0])
        xmap, ymap = np.meshgrid(xmap, ymap)
        points_z = self.depth
        median_depth = scipy.ndimage.median_filter(points_z, size=5)
        median_filter_error = np.absolute(points_z - median_depth)
        points_z[median_filter_error > 0.01] = np.nan
        points_x = (xmap - self.cx) / self.fx * points_z
        points_y = (ymap - self.cy) / self.fy * points_z

        points = np.stack((points_x, points_y, points_z), axis=2)
        return points

    def center_robot(self, bbox: Bbox):
        """
        Center the robots base and camera to face the center of the Object Bounding box
        """

        bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max = bbox

        bbox_center = [
            int((bbox_x_min + bbox_x_max) / 2),
            int((bbox_y_min + bbox_y_max) / 2),
        ]
        depth_obj = self.depth[bbox_center[1], bbox_center[0]]
        print(
            f"{self.query} height and depth: {((bbox_y_max - bbox_y_min) * depth_obj)/self.fy}, {depth_obj}"
        )

        # base movement
        dis = (bbox_center[0] - self.cx) / self.fx * depth_obj
        print(f"Base displacement {dis}")

        # camera tilt
        tilt = math.atan((bbox_center[1] - self.cy) / self.fy)
        print(f"Camera Tilt {tilt}")

        return [np.clip(-dis, -0.1, 0.1), self.head_tilt + tilt]

    def place(self, points: np.ndarray, seg_mask: np.ndarray) -> bool:
        points_x, points_y, points_z = points[:, :, 0], points[:, :, 1], points[:, :, 2]
        flat_x, flat_y, flat_z = (
            points_x.reshape(-1),
            -points_y.reshape(-1),
            -points_z.reshape(-1),
        )

        # Removing all points whose depth is zero(undetermined)
        zero_depth_seg_mask = (
            (flat_x != 0)
            * (flat_y != 0)
            * (flat_z != 0)
            * (~np.isnan(flat_z))
            * seg_mask.reshape(-1)
        )
        flat_x = flat_x[zero_depth_seg_mask]
        flat_y = flat_y[zero_depth_seg_mask]
        flat_z = flat_z[zero_depth_seg_mask]

        # 3d point cloud in camera orientation
        points1 = np.stack([flat_x, flat_y, flat_z], axis=-1)

        # Rotation matrix for camera tilt
        cam_to_3d_rot = np.array(
            [
                [1, 0, 0],
                [0, math.cos(self.head_tilt), math.sin(self.head_tilt)],
                [0, -math.sin(self.head_tilt), math.cos(self.head_tilt)],
            ]
        )

        # 3d point cloud with upright camera
        transformed_points = np.dot(points1, cam_to_3d_rot)

        # Removing floor points from point cloud
        floor_mask = transformed_points[:, 1] > -1.25
        transformed_points = transformed_points[floor_mask]
        transformed_x = transformed_points[:, 0]
        transformed_y = transformed_points[:, 1]
        transformed_z = transformed_points[:, 2]

        # Projected Median in the xz plane [parallel to floor]
        xz = np.stack([transformed_x * 100, transformed_z * 100], axis=-1).astype(int)
        unique_xz = np.unique(xz, axis=0)
        unique_xz_x, unique_xz_z = unique_xz[:, 0], unique_xz[:, 1]
        px, pz = np.median(unique_xz_x) / 100.0, np.median(unique_xz_z) / 100.0

        x_margin, z_margin = 0.1, 0
        x_mask = (transformed_x < (px + x_margin)) & (transformed_x > (px - x_margin))
        y_mask = (transformed_y < 0) & (transformed_y > -1.1)
        z_mask = (transformed_z < 0) & (transformed_z > (pz - z_margin))
        mask = x_mask & y_mask & z_mask
        py = np.max(transformed_y[mask])
        point = np.array([px, py, pz])  # Final placing point in upright camera coordinate system

        point[1] += 0.1
        transformed_point = cam_to_3d_rot @ point
        print(f"Placing point of Object relative to camera: {transformed_point}")

        return transformed_point

    def process(self, text, retry_flag, head_tilt=-1):
        """
        Wrapper for placing

        retry_flag 1 center robot
        retry_flag 2 placing
        """

        self.setup(text, head_tilt=head_tilt)

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        camera_image_file_name = self.save_dir + "/clean_" + str(self.tries) + ".jpg"
        print(f"Saving the camera image at {camera_image_file_name}")
        np.save(self.save_dir + "/depth_" + str(self.tries) + ".npy", self.depth)

        box_filename = f"{self.save_dir}/object_detection_{self.tries}.jpg"
        mask_filename = f"{self.save_dir}/semantic_segmentation_{self.tries}.jpg"

        # Object Segmentation Mask
        colors = np.array(self.image)
        colors[self.depth > self.max_depth + 0.3] = 1e-4
        colors = Image.fromarray(colors, "RGB")
        colors.save(camera_image_file_name)

        seg_mask, bbox = self.detection_model.detect_object(
            self.image,
            self.query,
            visualize_mask=True,
            box_filename=box_filename,
            mask_filename=mask_filename,
        )

        if bbox is None:
            print("Didn't detect the object, Trying Again")
            self.tries += 1
            print(f"Try no: {self.tries}")
            return None

        bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max = bbox

        # Center the robot
        if retry_flag == 1:
            self.tries += 1
            return self.center_robot(bbox)
        else:
            points = self.get_3d_points()
            self.tries = 1
            return self.place(points, seg_mask)
