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

import os

import launch
from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch_ros.actions import Node


def generate_launch_description():

    realsense_config = {
        "align_depth.enable": "True",
        "camera_name": "camera",
        "camera_namespace": "",
        # "decimation_filter.enable": "True",
        # "spatial_filter.enable": "True",
        # "temporal_filter.enable": "True",
        # "disparity_filter.enable": "False",
        "device_type": "d435i",
        "rgb_camera.color_profile": "640x480x30",
        "depth_module.depth_profile": "640x480x30",
        "depth_module.infra_profile": "640x480x30",
        "enable_gyro": "true",
        "enable_accel": "true",
        "gyro_fps": "200",
        "accel_fps": "100",
    }
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("stretch_ros2_bridge"), "launch/rs_launch.py")
        ),
        launch_arguments=realsense_config.items(),
    )

    realsense_d405_config = {
        "align_depth.enable": "true",
        "camera_name": "gripper_camera",
        "camera_namespace": "",
        # "decimation_filter.enable": "True",
        # "spatial_filter.enable": "True",
        # "temporal_filter.enable": "True",
        # "disparity_filter.enable": "True",
        "device_type": "d405",
        "rgb_camera.color_profile": "640x480x15",
        "depth_module.depth_profile": "640x480x15",
        "depth_module.color_profile": "640X480X15",
        # "rgb_camera.profile": "480x270x30",
        # "depth_module.profile": "480x270x30",
        "rgb_camera.enable_auto_exposure": "true",
        "gyro_fps": "200",
        "accel_fps": "100",
        "pointcloud.enable": "true",
        "pointcloud.stream_filter": "2",
        "pointcloud.stream_filter_index": "0",
        "allow_no_texture_points": "true",
        "enable_sync": "true",
    }
    realsense_d405_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("stretch_ros2_bridge"), "launch/rs_launch.py")
        ),
        launch_arguments=realsense_d405_config.items(),
    )

    ld = LaunchDescription(
        [
            realsense_launch,
            realsense_d405_launch,
        ]
    )
    return ld
