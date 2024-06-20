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
    realsense_d405_config = {
        "align_depth.enable": "true",
        "camera_name": "ee_camera",
        "camera_namespace": "",
        # "decimation_filter.enable": "True",
        # "spatial_filter.enable": "True",
        # "temporal_filter.enable": "True",
        # "disparity_filter.enable": "True",
        "device_type": "d405",
        "rgb_camera.profile": "640x480x15",
        "depth_module.profile": "640x480x15",
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
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("realsense2_camera"), "launch/rs_launch.py")
        ),
        launch_arguments=realsense_d405_config.items(),
    )
    ld = LaunchDescription(
        [
            base_slam_launch,
            realsense_launch,
        ]
    )

    return ld
