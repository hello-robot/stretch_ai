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

    stretch_driver_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("stretch_core"),
                "launch/stretch_driver.launch.py",
            )
        ),
        launch_arguments={"mode": "navigation", "broadcast_odom_tf": "True"}.items(),
    )

    stretch_cameras_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("stretch_ros2_bridge"),
                "launch/cameras.launch.py",
            )
        )
    )

    camera_pose_publisher_node = Node(
        package="stretch_ros2_bridge",
        executable="camera_pose_publisher",
        name="camera_pose_publisher",
        on_exit=launch.actions.Shutdown(),
    )

    odometry_publisher_node = Node(
        package="stretch_ros2_bridge",
        executable="odom_tf_publisher",
        name="odom_tf_publisher",
        on_exit=launch.actions.Shutdown(),
    )

    goto_controller_node = Node(
        package="stretch_ros2_bridge",
        executable="goto_controller",
        name="goto_controller",
        on_exit=launch.actions.Shutdown(),
    )

    lidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("stretch_core"), "launch/rplidar.launch.py")
        ),
    )

    ld = LaunchDescription(
        [
            stretch_driver_launch,
            stretch_cameras_launch,
            lidar_launch,
            camera_pose_publisher_node,
            goto_controller_node,
            odometry_publisher_node,
        ]
    )
    return ld
