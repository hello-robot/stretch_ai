# Copyright 2024 Hello Robot Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# licence information maybe found below, if so.

import argparse
import pprint as pp

import dex_teleop_parameters as dt
import goal_from_teleop as gt
import gripper_to_goal as gg
import loop_timer as lt
import numpy as np
import simple_ik as si
import webcam_teleop_interface as wt

if __name__ == "__main__":

    args = dt.get_arg_parser().parse_args()
    use_fastest_mode = args.fast
    manipulate_on_ground = args.ground
    left_handed = args.left
    using_stretch_2 = args.stretch_2
    slide_lift_range = args.slide_lift_range

    # The 'default', 'slow', 'fast', and 'max' options are defined by
    # Hello Robot. The 'fastest_stretch_2' option has been specially tuned for
    # this application.
    #
    # WARNING: 'fastest_stretch_*' have velocities and accelerations that exceed
    # the factory 'max' values defined by Hello Robot.
    if use_fastest_mode:
        if using_stretch_2:
            robot_speed = "fastest_stretch_2"
        else:
            robot_speed = "fastest_stretch_3"
    else:
        robot_speed = "slow"
    print("running with robot_speed =", robot_speed)

    lift_middle = dt.get_lift_middle(manipulate_on_ground)
    center_configuration = dt.get_center_configuration(lift_middle)
    starting_configuration = dt.get_starting_configuration(lift_middle)

    if left_handed:
        webcam_aruco_detector = wt.WebcamArucoDetector(
            tongs_prefix="left", visualize_detections=False
        )
    else:
        webcam_aruco_detector = wt.WebcamArucoDetector(
            tongs_prefix="right", visualize_detections=False
        )

    # Initialize IK
    simple_ik = si.SimpleIK()

    # Define the center position for the wrist that corresponds with
    # the teleop origin.
    center_wrist_position = simple_ik.fk_rotary_base(center_configuration)

    gripper_to_goal = gg.GripperToGoal(
        robot_speed, starting_configuration, dt.robot_allowed_to_move, using_stretch_2
    )

    goal_from_markers = gt.GoalFromMarkers(
        dt.teleop_origin, center_wrist_position, slide_lift_range=slide_lift_range
    )

    loop_timer = lt.LoopTimer()
    print_timing = False
    print_goal = False

    while True:
        loop_timer.start_of_iteration()
        markers = webcam_aruco_detector.process_next_frame()
        goal_dict = goal_from_markers.get_goal_dict(markers)
        if goal_dict:
            if print_goal:
                print("goal_dict =")
                pp.pprint(goal_dict)
            gripper_to_goal.update_goal(**goal_dict)
        loop_timer.end_of_iteration()
        if print_timing:
            loop_timer.pretty_print()


##############################################################
## NOTES
##############################################################

#######################################
#
# Overview
#
# Dexterous teleoperation uses a marker dictionary representing either
# a real or virtual ArUco marker specified with respect to the
# camera's frame of reference. The marker's position controls the
# robot's wrist position via inverse kinematics (IK). The marker's
# orientation directly controls the joints of the robot's dexterous
# wrist.
#
#######################################

#######################################
#
# The following coordinate systems are important to this teleoperation
# code
#
#######################################

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

#######################################
# Camera Coordinate System
#
# Camera on the floor looking with the top of the camer facing away
# from the person.
#
# This configuration matches the world frame's coordinate system with
# a different origin that is mostly just translated along the x and y
# axes.
#
# Origin likely at the optical cemter of a pinhole
# model of the camera.
#
# The descriptions below describe when the robot's mobile base is at
# theta = 0 deg.
#
# x-axis
# human left is pos / robot forward is pos
# human right is neg / robot backward is neg

# y-axis
# human arm extended is neg / robot arm extended is neg
# human arm retracted is pos / robot arm retracted is pos

# z-axis
# up is positive for person and the robot
# down is negative for person and the robot
#
#######################################

#######################################
# IK World Frame Coordinate System
#
# Origin at the axis of rotation of the mobile
# base on the floor.
#
# x-axis
# human/robot left is pos
# human/robot right is neg

# y-axis
# human/robot forward is neg
# human/robot backward is pos

# z-axis
# human/robot up is pos
# human/robot down is neg
#
#######################################

#######################################
# Robot Wrist Control

# wrist yaw
#     - : deployed direction
#     0 : straight out parallel to the telescoping arm
#     + : stowed direction

# wrist pitch
#     - : up
#     0 : horizontal
#     + : down

# wrist roll
#     - :
#     0 : horizontal
#     + :
#
#######################################

##############################################################
