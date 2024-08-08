# # Copyright (c) Hello Robot, Inc.
# #
# # This source code is licensed under the APACHE 2.0 license found in the
# # LICENSE file in the root directory of this source tree.
# #
# # Some code may be adapted from other open-source works with their respective licenses. Original
# # licence information maybe found below, if so.
#

# Copyright (c) Hello Robot, Inc.
#
# This source code is licensed under the APACHE 2.0 license found in the
# LICENSE file in the root directory of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# licence information maybe found below, if so.


import os

import launch
from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch_ros.actions import Node


def generate_launch_description():

    base_nodes = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("stretch_ros2_bridge"),
                "launch/start_ros_nodes.launch.py",
            )
        )
    )

    orbslam_node = Node(
        package="stretch_ros2_bridge",
        executable="orbslam3",
        name="orbslam3",
        on_exit=launch.actions.Shutdown(),
    )

    state_estimator_node = Node(
        package="stretch_ros2_bridge",
        executable="state_estimator",
        name="state_estimator",
        output="screen",
        on_exit=launch.actions.Shutdown(),
    )

    ld = LaunchDescription(
        [
            base_nodes,
            state_estimator_node,
            orbslam_node,
        ]
    )

    return ld
