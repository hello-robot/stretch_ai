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
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    stretch_navigation_path = get_package_share_directory("stretch_nav2")
    # start_robot_arg = DeclareLaunchArgument("start_robot", default_value="false")
    # rviz_arg = DeclareLaunchArgument("rviz", default_value="false")
    declare_use_sim_time_argument = DeclareLaunchArgument(
        "use_sim_time", default_value="false", description="Use simulation/Gazebo clock"
    )
    use_sim_time = LaunchConfiguration("use_sim_time")

    base_nodes = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("stretch_ros2_bridge"),
                "launch/start_ros_nodes.launch.py",
            )
        )
    )
    slam_params_file = LaunchConfiguration("slam_params_file")

    declare_slam_params_file_cmd = DeclareLaunchArgument(
        "slam_params_file",
        default_value=os.path.join(
            get_package_share_directory("stretch_nav2"),
            "config",
            "mapper_params_online_async.yaml",
        ),
        description="Full path to the ROS2 parameters file to use for the slam_toolbox node",
    )

    start_async_slam_toolbox_node = Node(
        parameters=[slam_params_file, {"use_sim_time": use_sim_time}],
        package="slam_toolbox",
        executable="async_slam_toolbox_node",
        name="slam_toolbox",
        output="screen",
        on_exit=launch.actions.Shutdown(),
    )

    state_estimator_node = Node(
        package="stretch_ros2_bridge",
        executable="state_estimator",
        name="state_estimator",
        output="screen",
        on_exit=launch.actions.Shutdown(),
    )

    orbslam_node = Node(
        package="stretch_ros2_bridge",
        executable="orbslam3",
        name="orbslam3",
        on_exit=launch.actions.Shutdown(),
    )

    ld = LaunchDescription(
        [
            base_nodes,
            declare_slam_params_file_cmd,
            state_estimator_node,
            orbslam_node,
        ]
    )

    ld.add_action(declare_use_sim_time_argument)
    ld.add_action(start_async_slam_toolbox_node)

    return ld
