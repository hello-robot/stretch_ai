# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import os

import numpy as np

import stretch.app.dex_teleop.dex_teleop_parameters as dt


class GoalFromMarkers:
    """Compute a goal based on the locations of detected aruco markers."""

    def __init__(self, teleop_origin, initial_center_wrist_position, slide_lift_range=False):

        print("GoalFromMarkers: teleop_origin =", teleop_origin)
        print(
            "GoalFromMarkers: initial_center_wrist_position =",
            initial_center_wrist_position,
        )

        self.slide_lift_range = slide_lift_range

        self.grip_pose_marker_name = "tongs"
        self.grip_width_marker_name = "tongs"
        self.teleop_origin = teleop_origin
        self.initial_center_wrist_position = np.array(initial_center_wrist_position)
        self.center_wrist_position = self.initial_center_wrist_position.copy()

        # Regions at the top and bottom of the allowable tongs range
        # are reserved for changing the range over which the lift is
        # operating. This sliding region enables a user to use the
        # lift's full range without restarting the code.
        self.sliding_region_height = dt.lift_sliding_region_height

        # The tongs are ignored when they are outside of this range.
        self.tongs_min_dist_from_camera = (
            dt.min_dist_from_camera_to_tongs - self.sliding_region_height
        )
        self.tongs_max_dist_from_camera = (
            dt.max_dist_from_camera_to_tongs + self.sliding_region_height
        )

        print(
            "GoalFromMarkers: self.tongs_min_dist_from_camera = {:.2f} cm".format(
                self.tongs_min_dist_from_camera * 100.0
            )
        )
        print(
            "GoalFromMarkers: self.tongs_max_dist_from_camera = {:.2f} cm".format(
                self.tongs_max_dist_from_camera * 100.0
            )
        )
        print(
            "GoalFromMarkers: self.sliding_region_height = {:.2f} cm".format(
                self.sliding_region_height * 100.0
            )
        )

        # These values determine when the lift range should be slid up
        # or down.
        self.slide_lift_range_down = self.tongs_min_dist_from_camera + self.sliding_region_height
        self.slide_lift_range_up = self.tongs_max_dist_from_camera - self.sliding_region_height

        # The tongs can be moved over a range of distances from the
        # camera to actively control the lift. This is the height of
        # this region when ignoring the lift's joint limits.
        self.tongs_lift_range = self.slide_lift_range_up - self.slide_lift_range_down
        print(
            "GoalFromMarkers: self.tongs_lift_range = {:.2f} cm".format(
                self.tongs_lift_range * 100.0
            )
        )

        # The maximum and minimum goal_wrist_position z values do not
        # need to be perfect due to joint limit checking performed by
        # the SimpleIK based on the specialized URDF joint
        # limits. They are specified with respect to the robot's
        # coordinate system.
        self.max_goal_wrist_position_z = dt.goal_max_position_z
        self.min_goal_wrist_position_z = dt.goal_min_position_z

        self.max_lift_range_offset = (
            (self.max_goal_wrist_position_z - self.center_wrist_position[2]) + self.teleop_origin[2]
        ) - (self.tongs_max_dist_from_camera - self.sliding_region_height)

        self.min_lift_range_offset = (
            (self.min_goal_wrist_position_z - self.center_wrist_position[2]) + self.teleop_origin[2]
        ) - (self.tongs_min_dist_from_camera + self.sliding_region_height)

        print("self.min_lift_range_offset = {:.2f} cm".format(self.min_lift_range_offset * 100.0))
        print("self.max_lift_range_offset = {:.2f} cm".format(self.max_lift_range_offset * 100.0))

        # Initialized the offset.
        self.lift_range_offset = 0.0

        # Set how fast the lift will be translated when being slid.
        self.lift_range_offset_change_per_timestep = dt.lift_range_offset_change_per_timestep

        self.in_sliding_region = False

        self.min_finger_width = dt.tongs_closed_grip_width
        self.max_finger_width = dt.tongs_open_grip_width

    def get_goal_dict(self, markers: dict) -> dict:
        """Takes in detected markers as a dict. Returns a dict with the goal pose for the wrist and the gripper width."""

        if markers:
            grip_pose_marker = markers.get(self.grip_pose_marker_name, None)
            grip_width_marker = markers.get(self.grip_width_marker_name, None)

            # If the marker (real or virtual) that specifies the
            # gripper pose goal has been observed and its z-axis
            # hasn't changed too much, proceed.
            if grip_pose_marker is not None:

                # Transform the gripper pose marker (real or
                # virtual) to a goal position for the wrist.

                # The wrist position goal is defined with respect
                # to the world frame, which has its origin where
                # the mobile base's rotational axis intersects
                # with the ground. The world frame's z-axis points
                # up. When the robot's mobile base has angle 0,
                # the world frame's x-axis points in front of the
                # robot and its y-axis points to the left of the
                # robot in the direction opposite to arm extension

                teleop_marker_position_in_camera_frame = grip_pose_marker["pos"]

                goal = None

                dist_from_camera = teleop_marker_position_in_camera_frame[2]

                tongs_at_valid_distance_from_camera = (
                    dist_from_camera > self.tongs_min_dist_from_camera
                ) and (dist_from_camera < self.tongs_max_dist_from_camera)

                if tongs_at_valid_distance_from_camera:

                    command_to_slide_lift_range = (
                        dist_from_camera < self.slide_lift_range_down
                    ) or (dist_from_camera > self.slide_lift_range_up)

                    if self.slide_lift_range and command_to_slide_lift_range:
                        if not self.in_sliding_region:
                            os.system("/usr/bin/canberra-gtk-play --id='bell'")
                            self.in_sliding_region = True
                        if dist_from_camera < self.slide_lift_range_down:
                            self.lift_range_offset = (
                                self.lift_range_offset - self.lift_range_offset_change_per_timestep
                            )
                            teleop_marker_position_in_camera_frame[2] = self.slide_lift_range_down
                            teleop_marker_position_in_camera_frame[2] = self.slide_lift_range_down
                        elif dist_from_camera > self.slide_lift_range_up:
                            self.lift_range_offset = (
                                self.lift_range_offset + self.lift_range_offset_change_per_timestep
                            )
                            teleop_marker_position_in_camera_frame[2] = self.slide_lift_range_up
                        self.lift_range_offset = min(
                            self.max_lift_range_offset, self.lift_range_offset
                        )
                        self.lift_range_offset = max(
                            self.min_lift_range_offset, self.lift_range_offset
                        )

                        print(
                            "self.lift_range_offset = {:.2f} cm".format(
                                self.lift_range_offset * 100.0
                            )
                        )
                    else:
                        if self.in_sliding_region:
                            os.system("/usr/bin/canberra-gtk-play --id='bell'")
                            self.in_sliding_region = False

                    goal_wrist_position = (
                        teleop_marker_position_in_camera_frame - self.teleop_origin
                    ) + self.center_wrist_position
                    goal_wrist_position[2] = goal_wrist_position[2] + self.lift_range_offset

                    # If the gripper width marker (virtual or real)
                    # has been observed, use it to command the robot's
                    # gripper.
                    goal_grip_width = None
                    if grip_width_marker is not None:
                        grip_width = grip_width_marker["info"]["grip_width"]
                        # convert to value between 0.0 and 1.0
                        goal_grip_width = (
                            np.clip(grip_width, self.min_finger_width, self.max_finger_width)
                            - self.min_finger_width
                        ) / (self.max_finger_width - self.min_finger_width)
                        # convert to value between -100.0 and 100.0

                    goal_x_axis = grip_pose_marker["x_axis"]
                    goal_y_axis = grip_pose_marker["y_axis"]
                    goal_z_axis = grip_pose_marker["z_axis"]

                    goal = {
                        "grip_width": goal_grip_width,
                        "wrist_position": goal_wrist_position,
                        "gripper_x_axis": goal_x_axis,
                        "gripper_y_axis": goal_y_axis,
                        "gripper_z_axis": goal_z_axis,
                    }

                return goal
            else:
                return None

    def get_goal_array(self, markers):
        dict_out = self.get_goal_dict(markers)
        return dt.goal_dict_to_array(dict_out)
