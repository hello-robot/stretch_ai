# Copyright (c) Hello Robot, Inc.
#
# This source code is licensed under the APACHE 2.0 license found in the
# LICENSE file in the root directory of this source tree.
# 
# Some code may be adapted from other open-source works with their respective licenses. Original
# licence information maybe found below, if so.


import os

from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch_ros.actions import Node


def generate_launch_description():
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
            base_slam_launch,
        ]
    )

    return ld
