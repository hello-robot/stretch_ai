# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import math
import time

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation
from yaml.loader import SafeLoader

import stretch.app.dex_teleop.dex_teleop_parameters as dt
import stretch.app.dex_teleop.teleop_aruco_detector as ad
import stretch.app.dex_teleop.webcam as wc
from stretch.utils.config import get_full_config_path


def pixel_from_3d(xyz, camera_info):
    x_in, y_in, z_in = xyz
    camera_matrix = camera_info["camera_matrix"]
    f_x = camera_matrix[0, 0]
    c_x = camera_matrix[0, 2]
    f_y = camera_matrix[1, 1]
    c_y = camera_matrix[1, 2]
    x_pix = ((f_x * x_in) / z_in) + c_x
    y_pix = ((f_y * y_in) / z_in) + c_y
    xy = np.array([x_pix, y_pix])
    return xy


def pixel_to_3d(xy_pix, z_in, camera_info):
    x_pix, y_pix = xy_pix
    camera_matrix = camera_info["camera_matrix"]
    f_x = camera_matrix[0, 0]
    c_x = camera_matrix[0, 2]
    f_y = camera_matrix[1, 1]
    c_y = camera_matrix[1, 2]
    x_out = ((x_pix - c_x) * z_in) / f_x
    y_out = ((y_pix - c_y) * z_in) / f_y
    xyz_out = np.array([x_out, y_out, z_in])
    return xyz_out


class WebcamArucoDetector:
    def __init__(
        self,
        tongs_prefix,
        visualize_detections=False,
        show_debug_images: bool = False,
        platform: str = "linux",
    ):

        # self.webcam = wc.Webcam(fps=30, image_width=800, image_height=448, use_calibration=True)
        self.webcam = wc.Webcam(
            fps=30, image_width=1920, image_height=1080, use_calibration=True, platform=platform
        )
        # self.webcam = wc.Webcam(fps=15, image_width=1920, image_height=1080, use_calibration=True)

        self.first_frame = True
        self.visualize_detections = visualize_detections

        self.marker_info = {}
        aruco_marker_info_file_name = get_full_config_path(
            "app/dex_teleop/teleop_aruco_marker_info_" + dt.tongs_to_use + ".yaml"
        )
        with open(aruco_marker_info_file_name) as f:
            self.marker_info = yaml.load(f, Loader=SafeLoader)

        if not self.marker_info:
            print(
                "WebcamArucoDetector: The ArUco file, "
                + aruco_marker_info_file_name
                + ", was not found, so no ArUco markers will be detected."
            )

        self.aruco_detector = ad.ArucoDetector(
            marker_info=self.marker_info, show_debug_images=show_debug_images
        )
        self.tongs_prefix = tongs_prefix

        # define marker names
        self.left_top_name = self.tongs_prefix + "_tongs_left_top"
        self.right_top_name = self.tongs_prefix + "_tongs_right_top"
        self.left_bottom_name = self.tongs_prefix + "_tongs_left_bottom"
        self.right_bottom_name = self.tongs_prefix + "_tongs_right_bottom"
        self.left_front_name = self.tongs_prefix + "_tongs_left_front"
        self.right_front_name = self.tongs_prefix + "_tongs_right_front"
        self.left_side_name = self.tongs_prefix + "_tongs_left_side"
        self.right_side_name = self.tongs_prefix + "_tongs_right_side"

        self.previous_grip_width = None

        self.debug_side_marker = False

        self.cube_side = dt.tongs_cube_side
        self.tongs_pin_joint_to_marker_center = dt.tongs_pin_joint_to_marker_center
        self.tongs_pin_joint_to_tong_tip = dt.tongs_pin_joint_to_tong_tip
        self.tongs_open_grip_width = dt.tongs_open_grip_width
        self.tongs_marker_center_to_tong_tip = dt.tongs_marker_center_to_tong_tip

    def process_tongs(self, markers):
        # create a grip width along with a single position and three
        # axes that are analogous to a single ArUco marker (e.g., the
        # toy cube)

        #######################################
        # ArUco Coordinate System
        #
        # Origin in the middle of the ArUco marker.
        #
        # x-axis
        # right side when looking at marker is pos
        # left side when looking at marker is neg

        # y-axis
        # top of marker is pos
        # bottom of marker is neg

        # z-axis
        # normal to marker surface is pos
        # pointing into the marker surface is neg
        #
        #######################################

        virtual_marker = None

        tongs_left_top_marker = markers.get(self.left_top_name, None)
        tongs_right_top_marker = markers.get(self.right_top_name, None)

        tongs_left_bottom_marker = markers.get(self.left_bottom_name, None)
        tongs_right_bottom_marker = markers.get(self.right_bottom_name, None)

        tongs_left_front_marker = markers.get(self.left_front_name, None)
        tongs_right_front_marker = markers.get(self.right_front_name, None)

        tongs_left_side_marker = markers.get(self.left_side_name, None)
        tongs_right_side_marker = markers.get(self.right_side_name, None)

        top_visible = (tongs_left_top_marker is not None) and (tongs_right_top_marker is not None)
        bottom_visible = (tongs_left_bottom_marker is not None) and (
            tongs_right_bottom_marker is not None
        )
        front_visible = (tongs_left_front_marker is not None) and (
            tongs_right_front_marker is not None
        )
        left_side_visible = tongs_left_side_marker is not None
        right_side_visible = tongs_right_side_marker is not None

        main_markers_visible = bottom_visible or front_visible or top_visible

        if not main_markers_visible:

            if left_side_visible or right_side_visible:
                # this is an approximation
                if self.previous_grip_width is not None:
                    grip_width = self.previous_grip_width
                else:
                    grip_width = self.tongs_open_grip_width

                # Constant angle from the right triangle formed by the tongs pin joint, the bottom ArUco marker center, and the tong tip.
                marker_center_to_tong_tip = self.tongs_marker_center_to_tong_tip
                opposite_side = marker_center_to_tong_tip
                adjacent_side = self.tongs_pin_joint_to_tong_tip
                constant_angle = math.atan(opposite_side / adjacent_side)
                if self.debug_side_marker:
                    print()
                    print("----------------------------------------")
                    print("Only side marker visible.")
                    print()
                    print("grip_width = {:.2f} cm".format(grip_width * 100.0))
                    print(
                        "marker_center_to_tong_tip = {:.2f} cm".format(
                            marker_center_to_tong_tip * 100.0
                        )
                    )
                    print(
                        "tongs_pin_joint_to_tong_tip = {:.2f} cm".format(
                            self.tongs_pin_joint_to_tong_tip * 100.0
                        )
                    )
                    print("constant_angle = {:.2f} deg".format(180.0 * (constant_angle / np.pi)))
                    print()

                # Variable angle from the isosceles triangle formed by the centers of the bottom ArUco markers and the tongs pin joint.
                distance_between_markers = grip_width
                hypotenuse = self.tongs_pin_joint_to_marker_center
                opposite_side = distance_between_markers / 2.0
                variable_angle = math.asin(opposite_side / hypotenuse)
                if self.debug_side_marker:
                    print(
                        "tongs_pin_joint_to_marker_cecnter = {:.2f} cm".format(
                            self.tongs_pin_joint_to_marker_center * 100.0
                        )
                    )
                    print(
                        "distance_between_markers = {:.2f} cm".format(
                            distance_between_markers * 100.0
                        )
                    )
                    print("variable_angle = {:.2f} deg".format(180.0 * (variable_angle / np.pi)))
                    print()

                # Find angular correction to find tongs marker orientation from a single side
                angle_correction = variable_angle - constant_angle
                if self.debug_side_marker:
                    print(
                        "angle_correction = {:.2f} deg".format(180.0 * (angle_correction / np.pi))
                    )

                if left_side_visible:
                    grip_pos = tongs_left_side_marker["pos"].copy()
                    grip_z_axis = -tongs_left_side_marker["y_axis"].copy()
                    grip_y_axis = -tongs_left_side_marker["x_axis"].copy()

                    rotvec = angle_correction * grip_z_axis
                    r = Rotation.from_rotvec(rotvec)
                    grip_y_axis = r.apply(grip_y_axis)

                    grip_x_axis = np.cross(grip_y_axis, grip_z_axis)

                    # this is an approximation
                    grip_pos = (
                        grip_pos
                        + (-(grip_width / 2.0) * grip_x_axis)
                        + (-(self.cube_side / 2.0) * tongs_left_side_marker["y_axis"])
                        + (-(self.cube_side / 2.0) * tongs_left_side_marker["z_axis"])
                    )

                elif right_side_visible:
                    grip_pos = tongs_right_side_marker["pos"].copy()
                    grip_z_axis = -tongs_right_side_marker["y_axis"].copy()
                    grip_y_axis = tongs_right_side_marker["x_axis"].copy()

                    rotvec = -angle_correction * grip_z_axis
                    r = Rotation.from_rotvec(rotvec)
                    grip_y_axis = r.apply(grip_y_axis)

                    grip_x_axis = np.cross(grip_y_axis, grip_z_axis)

                    # this is an approximation
                    grip_pos = (
                        grip_pos
                        + ((grip_width / 2.0) * grip_x_axis)
                        + (-(self.cube_side / 2.0) * tongs_right_side_marker["y_axis"])
                        + (-(self.cube_side / 2.0) * tongs_right_side_marker["z_axis"])
                    )

                virtual_marker_name = "tongs"
                virtual_marker = {
                    "name": virtual_marker_name,
                    "pos": grip_pos,
                    "x_axis": grip_x_axis,
                    "y_axis": grip_y_axis,
                    "z_axis": grip_z_axis,
                    "info": {"name": virtual_marker_name, "grip_width": grip_width},
                }

                self.previous_grip_width = grip_width

        else:
            if bottom_visible:
                left_bottom_min_dist = tongs_left_bottom_marker["min_dist_between_corners"]
                right_bottom_min_dist = tongs_left_bottom_marker["min_dist_between_corners"]
                bottom_min_dist = min(left_bottom_min_dist, right_bottom_min_dist)
            else:
                bottom_min_dist = -1.0

            if front_visible:
                left_front_min_dist = tongs_left_front_marker["min_dist_between_corners"]
                right_front_min_dist = tongs_left_front_marker["min_dist_between_corners"]
                front_min_dist = min(left_front_min_dist, right_front_min_dist)
            else:
                front_min_dist = -1.0

            if top_visible:
                left_top_min_dist = tongs_left_top_marker["min_dist_between_corners"]
                right_top_min_dist = tongs_left_top_marker["min_dist_between_corners"]
                top_min_dist = min(left_top_min_dist, right_top_min_dist)
            else:
                top_min_dist = -1.0

            if (bottom_min_dist >= front_min_dist) and (bottom_min_dist >= top_min_dist):
                # Best visibility is for the bottom markers

                left_pos = tongs_left_bottom_marker["pos"].copy()
                left_y_axis = tongs_left_bottom_marker["y_axis"].copy()
                left_x_axis = tongs_left_bottom_marker["x_axis"].copy()

                right_pos = tongs_right_bottom_marker["pos"].copy()
                right_y_axis = tongs_right_bottom_marker["y_axis"].copy()
                right_x_axis = tongs_right_bottom_marker["x_axis"].copy()

                grip_width = np.linalg.norm(right_pos - left_pos)

                grip_pos = (left_pos + right_pos) / 2.0

                # axis's have unit length, so this should provide the bisector
                x_bisector = left_x_axis + right_x_axis
                x_bisector_length = np.linalg.norm(x_bisector)
                if x_bisector_length > 0.01:
                    grip_x_axis = x_bisector / x_bisector_length
                else:
                    print(
                        "WebcamArucoDetector.process_tongs: bottom marker x_bisector_length <= 0.01 so not returning a marker. x_bisector_length =",
                        x_bisector_length,
                    )
                    return None

                # axis's have unit length, so this should provide the bisector
                y_bisector = left_y_axis + right_y_axis
                y_bisector_length = np.linalg.norm(y_bisector)
                if y_bisector_length > 0.01:
                    grip_y_axis = y_bisector / y_bisector_length
                else:
                    print(
                        "WebcamArucoDetector.process_tongs: bottom marker y_bisector_length <= 0.01 so not returning a marker. y_bisector_length =",
                        y_bisector_length,
                    )
                    return None

                # Now, modify y axis to make it orthogonal to y axis.
                x_projection = grip_x_axis.dot(grip_y_axis)
                grip_y_axis = grip_y_axis - x_projection
                grip_y_axis = grip_y_axis / np.linalg.norm(grip_y_axis)

                grip_z_axis = np.cross(grip_x_axis, grip_y_axis)

            elif front_min_dist >= top_min_dist:
                # Best visibility is for the front markers

                left_pos = tongs_left_front_marker["pos"].copy()
                left_z_axis = tongs_left_front_marker["z_axis"].copy()
                left_y_axis = tongs_left_front_marker["y_axis"].copy()
                left_x_axis = tongs_left_front_marker["x_axis"].copy()

                right_pos = tongs_right_front_marker["pos"].copy()
                right_z_axis = tongs_right_front_marker["z_axis"].copy()
                right_y_axis = tongs_right_front_marker["y_axis"].copy()
                right_x_axis = tongs_right_front_marker["x_axis"].copy()

                # Adjust positions to match bottom marker positions
                half_cube_side = self.cube_side / 2.0
                bottom_left_pos = (
                    left_pos - (half_cube_side * left_z_axis) - (half_cube_side * left_y_axis)
                )
                bottom_right_pos = (
                    right_pos - (half_cube_side * right_z_axis) - (half_cube_side * right_y_axis)
                )

                grip_width = np.linalg.norm(bottom_right_pos - bottom_left_pos)

                grip_pos = (bottom_left_pos + bottom_right_pos) / 2.0

                # axis's have unit length, so this should provide the bisector
                x_bisector = left_x_axis + right_x_axis
                x_bisector_length = np.linalg.norm(x_bisector)
                if x_bisector_length > 0.01:
                    grip_x_axis = x_bisector / x_bisector_length
                else:
                    print(
                        "WebcamArucoDetector.process_tongs: bottom marker x_bisector_length <= 0.01 so not returning a marker. x_bisector_length =",
                        x_bisector_length,
                    )
                    return None

                # axis's have unit length, so this should provide the bisector
                z_bisector = -(left_y_axis + right_y_axis)
                z_bisector_length = np.linalg.norm(z_bisector)
                if z_bisector_length > 0.01:
                    grip_z_axis = z_bisector / z_bisector_length
                else:
                    print(
                        "WebcamArucoDetector.process_tongs: front marker bisector_length <= 0.01 so not returning a marker. bisector_length =",
                        z_bisector_length,
                    )
                    return None

                # Now, modify y axis to make it orthogonal to y axis.
                x_projection = grip_x_axis.dot(grip_z_axis)
                grip_z_axis = grip_z_axis - x_projection
                grip_z_axis = grip_z_axis / np.linalg.norm(grip_z_axis)

                grip_y_axis = np.cross(grip_z_axis, grip_x_axis)
            else:
                # Best visibility is for the top markers

                left_pos = tongs_left_top_marker["pos"].copy()
                left_y_axis = tongs_left_top_marker["y_axis"].copy()
                left_x_axis = tongs_left_top_marker["x_axis"].copy()
                left_z_axis = tongs_left_top_marker["z_axis"].copy()

                right_pos = tongs_right_top_marker["pos"].copy()
                right_y_axis = tongs_right_top_marker["y_axis"].copy()
                right_x_axis = tongs_right_top_marker["x_axis"].copy()
                right_z_axis = tongs_right_top_marker["z_axis"].copy()

                grip_width = np.linalg.norm(right_pos - left_pos)

                # Transform the markers to be like the bottom markers

                # Adjust positions to match bottom marker positions
                bottom_left_pos = left_pos - (self.cube_side * left_z_axis)
                bottom_right_pos = right_pos - (self.cube_side * right_z_axis)
                grip_pos = (bottom_left_pos + bottom_right_pos) / 2.0

                # Invert the z axes and x axes
                left_z_axis = -left_z_axis
                right_z_axis = -right_z_axis

                left_x_axis = -left_x_axis
                right_x_axis = -right_x_axis

                # axis's have unit length, so this should provide the bisector
                x_bisector = left_x_axis + right_x_axis
                x_bisector_length = np.linalg.norm(x_bisector)
                if x_bisector_length > 0.01:
                    grip_x_axis = x_bisector / x_bisector_length
                else:
                    print(
                        "WebcamArucoDetector.process_tongs: bottom marker x_bisector_length <= 0.01 so not returning a marker. x_bisector_length =",
                        x_bisector_length,
                    )
                    return None

                # axis's have unit length, so this should provide the bisector
                y_bisector = left_y_axis + right_y_axis
                y_bisector_length = np.linalg.norm(y_bisector)
                if y_bisector_length > 0.01:
                    grip_y_axis = y_bisector / y_bisector_length
                else:
                    print(
                        "WebcamArucoDetector.process_tongs: bottom marker y_bisector_length <= 0.01 so not returning a marker. y_bisector_length =",
                        y_bisector_length,
                    )
                    return None

                # Now, modify y axis to make it orthogonal to y axis.
                x_projection = grip_x_axis.dot(grip_y_axis)
                grip_y_axis = grip_y_axis - x_projection
                grip_y_axis = grip_y_axis / np.linalg.norm(grip_y_axis)

                grip_z_axis = np.cross(grip_x_axis, grip_y_axis)

            # Move the position from the point between the two bottom
            # markers to the pin joint of the tongs at the tong's
            # midpoint in height.

            # Translate the position up along the negative z-axis
            grip_pos = grip_pos + (-(self.cube_side / 2.0) * grip_z_axis)

            # Translate the position back along the negative y-axis
            if grip_width is not None:
                distance_between_markers = grip_width
            elif self.prev_grip_width is not None:
                distance_between_markers = self.previous_grip_width
            else:
                distance_between_markers = self.tongs_open_grip_width
            hypotenuse = self.tongs_pin_joint_to_marker_center
            opposite_side = distance_between_markers / 2.0
            try:
                adjacent_side = math.sqrt(
                    (hypotenuse * hypotenuse) - (opposite_side * opposite_side)
                )
            except ValueError:
                print(
                    "WebcamArucoDetector.process_tongs: ValueError: hypotenuse =",
                    hypotenuse,
                    "opposite_side =",
                    opposite_side,
                )
                return None
            grip_pos = grip_pos + (-(adjacent_side) * grip_y_axis)

            virtual_marker_name = "tongs"
            virtual_marker = {
                "name": virtual_marker_name,
                "pos": grip_pos,
                "x_axis": grip_x_axis,
                "y_axis": grip_y_axis,
                "z_axis": grip_z_axis,
                "info": {"name": virtual_marker_name, "grip_width": grip_width},
            }

            self.previous_grip_width = grip_width

        return virtual_marker

    def process_next_frame(self):
        """Get next frame from webcam and use it to detect markers."""

        color_image, color_camera_info = self.webcam.get_next_frame()

        if color_image is None:
            return None, None

        self.aruco_detector.update(color_image, color_camera_info)
        markers = self.aruco_detector.get_detected_markers()

        if markers is None:
            return None, color_image

        virtual_tongs_marker = self.process_tongs(markers)
        if virtual_tongs_marker is not None:
            markers[virtual_tongs_marker["name"]] = virtual_tongs_marker

        if self.visualize_detections:
            cv2.imshow("ArUco Detections", color_image)
            cv2.waitKey(1)

        return markers, color_image


if __name__ == "__main__":
    print("cv2.__path__ =", cv2.__path__)
    webcam_aruco_detector = WebcamArucoDetector("right", visualize_detections=True)
    start_time = time.time()
    iterations = 0
    while True:
        markers = webcam_aruco_detector.process_next_frame()
        if markers:
            print("********************")
            print("markers =", markers)
            print("********************")
        iterations = iterations + 1
        current_time = time.time()
        total_duration = current_time - start_time
        average_period = total_duration / iterations
        average_frequency = 1.0 / average_period
        print()
        print("--- TIMING FOR ITERATIONS WITH ROBOT MOTION ---")
        print("number of iterations with robot motion =", iterations)
        print("average period =", "{:.2f}".format(average_period * 1000.0), "ms")
        print("average frequency =", "{:.2f}".format(average_frequency), "Hz")
        print("-----------------------------------------------")
