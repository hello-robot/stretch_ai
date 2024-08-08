# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import abc
from typing import Tuple

import numpy as np
import pyrealsense2 as rs


class Realsense(abc.ABC):
    def __init__(self, exposure: str = "auto"):
        self.exposure = exposure
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.profile = self.pipeline.start(self.config)

    def get_depth_scale(self) -> float:
        """Get scaling between depth values and metric units (meters)"""
        depth_sensor = self.profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        return depth_scale

    def get_images(self) -> Tuple[np.ndarray, np.ndarray]:
        depth_frame, color_frame = self.get_frames()
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        return depth_image, color_image

    def get_message(self) -> dict:
        """Get a message that can be sent via ZMQ"""
        depth_camera_info, color_camera_info = self.get_camera_infos()
        depth_scale = self.get_depth_scale()
        depth_image, color_image = self.get_images()
        realsense_output = {
            "depth_camera_info": depth_camera_info,
            "color_camera_info": color_camera_info,
            "depth_scale": depth_scale,
            "depth_image": depth_image,
            "color_image": color_image,
        }
        return realsense_output

    def wait_for_frames(self):
        return self.pipeline.wait_for_frames()

    def get_frames(self):
        frames = self.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        return depth_frame, color_frame

    def read_camera_infos(self):
        color_frame, depth_frame = self.get_frames()
        return get_camera_info(depth_frame), get_camera_info(color_frame)

    def get_camera_infos(self):
        raise NotImplementedError()


def get_camera_info(frame):
    """Get camera info for a realsense"""
    intrinsics = rs.video_stream_profile(frame.profile).get_intrinsics()

    # from Intel's documentation
    # https://intelrealsense.github.io/librealsense/python_docs/_generated/pyrealsense2.intrinsics.html#pyrealsense2.intrinsics
    # "
    # coeffs	Distortion coefficients
    # fx	Focal length of the image plane, as a multiple of pixel width
    # fy	Focal length of the image plane, as a multiple of pixel height
    # height	Height of the image in pixels
    # model	Distortion model of the image
    # ppx	Horizontal coordinate of the principal point of the image, as a pixel offset from the left edge
    # ppy	Vertical coordinate of the principal point of the image, as a pixel offset from the top edge
    # width	Width of the image in pixels
    # "

    # out = {
    #     'dist_model' : intrinsics.model,
    #     'dist_coeff' : intrinsics.coeffs,
    #     'fx' : intrinsics.fx,
    #     'fy' : intrinsics.fy,
    #     'height' : intrinsics.height,
    #     'width' : intrinsics.width,
    #     'ppx' : intrinsics.ppx,
    #     'ppy' : intrinsics.ppy
    #     }

    camera_matrix = np.array(
        [
            [intrinsics.fx, 0.0, intrinsics.ppx],
            [0.0, intrinsics.fy, intrinsics.ppy],
            [0.0, 0.0, 1.0],
        ]
    )

    distortion_model = intrinsics.model

    distortion_coefficients = np.array(intrinsics.coeffs)

    camera_info = {
        "camera_matrix": camera_matrix,
        "distortion_coefficients": distortion_coefficients,
        "distortion_model": distortion_model,
    }

    return camera_info
