import os

import launch
from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    start_server = Node(
        package="stretch_ros2_bridge",
        executable="server",
        name="ros2_zmq_server",
        output="screen",
        on_exit=launch.actions.Shutdown(),
    )

    base_slam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("stretch_ros2_bridge"),
                "launch/startup_stetch_hector_slam.launch.py",
            )
        )
    )
    ld = LaunchDescription(
        [
            base_slam_launch,
            start_server,
        ]
    )

    return ld
