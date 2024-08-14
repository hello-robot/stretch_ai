# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import os

import launch
from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():

    orbslam_node = Node(
        package="stretch_ros2_bridge",
        executable="orbslam3",
        name="orbslam3",
        on_exit=launch.actions.Shutdown(),
    )

    ld = LaunchDescription(
        [
            orbslam_node,
        ]
    )

    return ld
