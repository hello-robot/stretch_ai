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
    start_server = Node(
        package="stretch_ros2_bridge",
        executable="server_no_d405",
        name="ros2_zmq_server",
        output="screen",
        on_exit=launch.actions.Shutdown(),
    )

    stretch_cameras_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("stretch_ros2_bridge"),
                "launch/cameras_d435_only.launch.py",
            )
        )
    )

    base_slam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("stretch_ros2_bridge"),
                "launch/startup_stretch_hector_slam.launch.py",
            )
        )
    )

    ld = LaunchDescription(
        [
            stretch_cameras_launch,
            base_slam_launch,
            start_server,
        ]
    )

    return ld
