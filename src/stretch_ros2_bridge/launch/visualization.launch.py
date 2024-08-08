import os

from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import TextSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    image_rotation_node = Node(
        package="stretch_ros2_bridge",
        executable="rotate_images",
        name="rotate_images_from_stretch_head",
    )

    return LaunchDescription([image_rotation_node])
