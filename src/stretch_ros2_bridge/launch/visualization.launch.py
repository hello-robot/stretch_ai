# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.


from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    image_rotation_node = Node(
        package="stretch_ros2_bridge",
        executable="rotate_images",
        name="rotate_images_from_stretch_head",
    )

    return LaunchDescription([image_rotation_node])
