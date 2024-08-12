#!/usr/bin/env python3

# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.


import cv2
import cv2.aruco as aruco
import numpy as np


def minimum_distance_between_corners(corners):
    # calculate the 6 distances between the corners and return the minimum
    c0 = corners[0]
    dist0 = np.min(np.linalg.norm(corners[1:4] - c0, axis=1))
    c1 = corners[1]
    dist1 = np.min(np.linalg.norm(corners[2:4] - c1, axis=1))
    c2 = corners[2]
    dist2 = np.min(np.linalg.norm(corners[3:4] - c2, axis=1))
    return np.min(np.array([dist0, dist1, dist2]))


class ArucoMarker:
    def __init__(self, aruco_id, marker_info, show_debug_images=False):
        self.show_debug_images = show_debug_images

        self.aruco_id = aruco_id
        colormap = cv2.COLORMAP_HSV
        offset = 0
        i = (offset + (self.aruco_id * 29)) % 255
        image = np.uint8([[[i]]])
        id_color_image = cv2.applyColorMap(image, colormap)
        bgr = id_color_image[0, 0]
        self.id_color = [bgr[2], bgr[1], bgr[0]]

        self.frame_id = "camera_color_optical_frame"
        self.info = marker_info.get(str(self.aruco_id), None)

        if self.info is None:
            self.info = marker_info["default"]
        self.length_of_marker_mm = self.info["length_mm"]
        self.use_rgb_only = self.info["use_rgb_only"]

        self.frame_number = None
        self.ready = False
        self.x_axis = None
        self.y_axis = None
        self.z_axis = None
        self.min_dist_between_corners = None

    def update(self, corners, frame_number, rgb_camera_info):
        self.corners = corners
        self.frame_number = frame_number

        self.rgb_camera_info = rgb_camera_info
        self.camera_matrix = self.rgb_camera_info["camera_matrix"]
        self.distortion_coefficients = self.rgb_camera_info["distortion_coefficients"]

        rvecs = np.zeros((1, 1, 3), dtype=np.float64)
        tvecs = np.zeros((1, 1, 3), dtype=np.float64)
        points_3D = np.array(
            [
                (-self.length_of_marker_mm / 2, self.length_of_marker_mm / 2, 0),
                (self.length_of_marker_mm / 2, self.length_of_marker_mm / 2, 0),
                (self.length_of_marker_mm / 2, -self.length_of_marker_mm / 2, 0),
                (-self.length_of_marker_mm / 2, -self.length_of_marker_mm / 2, 0),
            ]
        )

        unknown_variable, rvecs_ret, tvecs_ret = cv2.solvePnP(
            objectPoints=points_3D,
            imagePoints=self.corners,
            cameraMatrix=self.camera_matrix,
            distCoeffs=self.distortion_coefficients,
        )
        rvecs[0][:] = np.transpose(rvecs_ret)
        tvecs[0][:] = np.transpose(tvecs_ret)
        self.aruco_rotation = rvecs[0][0]

        # Convert ArUco position estimate to be in meters.
        self.aruco_position = tvecs[0][0] / 1000.0

        # TODO: do we need this depth estimate?
        # aruco_depth_estimate = self.aruco_position[2]

        self.marker_position = self.aruco_position
        R = np.identity(4)
        R[:3, :3] = cv2.Rodrigues(self.aruco_rotation)[0]
        self.x_axis = R[:3, 0]
        self.y_axis = R[:3, 1]
        self.z_axis = R[:3, 2]

        self.ready = True

    def get_min_dist_between_corners(self):
        return minimum_distance_between_corners(self.corners)

    def get_position_and_axes(self):
        # return copies of the position and axes
        pos = np.array(self.marker_position)
        x_axis = np.array(self.x_axis)
        y_axis = np.array(self.y_axis)
        z_axis = np.array(self.z_axis)
        return pos, x_axis, y_axis, z_axis

    def get_info(self):
        # return copy of marker_info
        return self.info.copy()

    def get_marker_poly(self, corners):
        poly_points = np.array(corners)
        poly_points = np.round(poly_points).astype(np.int32)
        return poly_points

    def draw_marker_poly(self, image):
        poly_points = self.get_marker_poly()
        cv2.fillConvexPoly(image, poly_points, (255, 0, 0))


class ArucoMarkerCollection:
    def __init__(self, marker_info, show_debug_images=False):
        self.show_debug_images = show_debug_images

        self.marker_info = marker_info
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        self.aruco_detection_parameters = aruco.DetectorParameters()
        # Apparently available in OpenCV 3.4.1, but not OpenCV 3.2.0.
        # self.aruco_detection_parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        self.aruco_detection_parameters.cornerRefinementMethod = aruco.CORNER_REFINE_APRILTAG
        # self.aruco_detection_parameters.cornerRefinementWinSize = 2
        self.collection = {}
        self.detector = aruco.ArucoDetector(self.aruco_dict, self.aruco_detection_parameters)
        self.frame_number = 0

        # Configuring slightly faster teleop
        self.detector_params = self.detector.getDetectorParameters()
        self.detector_params.useAruco3Detection = True
        self.detector.setDetectorParameters(self.detector_params)

    def __iter__(self):
        # iterates through currently visible ArUco markers
        keys = self.collection.keys()
        for k in keys:
            marker = self.collection[k]
            if marker.frame_number == self.frame_number:
                yield marker

    def draw_markers(self, image):
        return aruco.drawDetectedMarkers(image, self.aruco_corners, self.aruco_ids)

    def update(self, rgb_image, rgb_camera_info, verbose: bool = False):
        self.frame_number += 1
        self.rgb_image = rgb_image
        self.rgb_camera_info = rgb_camera_info
        self.gray_image = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2GRAY)
        image_height, image_width = self.gray_image.shape
        (
            self.aruco_corners,
            self.aruco_ids,
            aruco_rejected_image_points,
        ) = self.detector.detectMarkers(self.gray_image)
        if self.aruco_ids is None:
            num_detected = 0
        else:
            num_detected = len(self.aruco_ids)

        if self.aruco_ids is not None:
            for corners, aruco_id in zip(self.aruco_corners, self.aruco_ids):
                aruco_id = int(aruco_id)
                marker = self.collection.get(aruco_id, None)
                if marker is None:
                    new_marker = ArucoMarker(aruco_id, self.marker_info, self.show_debug_images)
                    self.collection[aruco_id] = new_marker

                self.collection[aruco_id].update(
                    corners[0], self.frame_number, self.rgb_camera_info
                )

        if verbose:
            print("Detected", num_detected, "markers")


class ArucoDetector:
    def __init__(self, marker_info=None, show_debug_images=False):
        self.rgb_image = None
        self.camera_info = None
        self.all_points = []
        self.show_debug_images = show_debug_images
        self.publish_marker_point_clouds = False
        self.marker_info = marker_info

        if self.marker_info is None:
            self.marker_info = {}

        self.aruco_marker_collection = ArucoMarkerCollection(
            self.marker_info, self.show_debug_images
        )

    def update(self, rgb_image, rgb_camera_info):
        self.rgb_image = rgb_image
        self.rgb_camera_info = rgb_camera_info

        self.aruco_marker_collection.update(self.rgb_image, self.rgb_camera_info)

        # save rotation for last
        if self.show_debug_images:
            aruco_image = self.aruco_marker_collection.draw_markers(self.rgb_image)
            display_aruco_image = cv2.rotate(aruco_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.imshow("Detected ArUco Markers", display_aruco_image)
            cv2.imshow("Detected ArUco Markers", aruco_image)
            cv2.waitKey(2)

    def get_detected_marker_dict(self):
        out = {}
        for m in self.aruco_marker_collection:
            aruco_id = m.aruco_id
            pos, x_axis, y_axis, z_axis = m.get_position_and_axes()
            min_dist_between_corners = m.get_min_dist_between_corners()
            info = m.get_info()
            out[aruco_id] = {
                "pos": pos,
                "x_axis": x_axis,
                "y_axis": y_axis,
                "z_axis": z_axis,
                "min_dist_between_corners": min_dist_between_corners,
                "info": info,
            }
        return out

    def get_detected_markers(self):
        markers = self.get_detected_marker_dict()

        # This changes keys to be marker names to make code less
        # sensitive to marker changes. Ideally, only the ArUco
        # detection code, as informed by the YAML file, cares about
        # the marker numbers.
        new_markers = {}
        for marker_num in markers.keys():
            m = markers[marker_num]
            m["info"]["marker_id"] = marker_num
            marker_name = m["info"]["name"]
            new_markers[marker_name] = m
        return new_markers


def get_special_frames(marker_dict):
    # only find origins of the special frames via translation
    # rpy rotation not implemented, yet
    info = marker_dict["info"]
    frames = info.get("frames")
    out = {}
    if frames is not None:
        marker_pos = marker_dict["pos"]
        marker_x_axis = marker_dict["x_axis"]
        marker_y_axis = marker_dict["y_axis"]
        marker_z_axis = marker_dict["z_axis"]
        for k in frames:
            t = frames[k]["trans"]
            # TODO: do we want to save rpy?
            # rpy = frames[k]["rpy"]
            frame_pos = (
                marker_pos
                + (t[0] * marker_x_axis)
                + (t[1] * marker_y_axis)
                + (t[2] * marker_z_axis)
            )
            frame_x_axis = np.copy(marker_x_axis)
            frame_y_axis = np.copy(marker_y_axis)
            frame_z_axis = np.copy(marker_z_axis)
            out[k] = {
                "pos": frame_pos,
                "x_axis": frame_x_axis,
                "y_axis": frame_y_axis,
                "z_axis": frame_z_axis,
            }
    return out


def main(args=None):
    detector = ArucoDetector()
    print("detector =", detector)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
