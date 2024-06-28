import os

import launch
from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    stretch_navigation_path = get_package_share_directory("stretch_nav2")
    # start_robot_arg = DeclareLaunchArgument("start_robot", default_value="false")
    # rviz_arg = DeclareLaunchArgument("rviz", default_value="false")

    declare_use_sim_time_argument = DeclareLaunchArgument(
        "use_sim_time", default_value="false", description="Use simulation/Gazebo clock"
    )

    stretch_driver_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("stretch_core"),
                "launch/stretch_driver.launch.py",
            )
        ),
        launch_arguments={"mode": "navigation", "broadcast_odom_tf": "True"}.items(),
    )

    realsense_config = {
        "align_depth.enable": "True",
        "camera_name": "camera",
        "camera_namespace": "",
        "device_type": "d435i",
        "depth_module.depth_profile": "640x480x15",
        "enable_depth": "true",
        "rgb_camera.color_profile": "640x480x15",
        "depth_module.infra_profile": "640x480x15",
        "enable_color": "true",
        "enable_gyro": "true",
        "enable_accel": "true",
        "gyro_fps": "200",
        "accel_fps": "100",
    }
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("realsense2_camera"), "launch/rs_launch.py")
        ),
        launch_arguments=realsense_config.items(),
    )

    declare_use_sim_time_argument = DeclareLaunchArgument(
        "use_sim_time", default_value="false", description="Use simulation/Gazebo clock"
    )

    # camera_pose_publisher_node = Node(
    #     package="stretch_ros2_bridge",
    #     executable="camera_pose_publisher",
    #     name="camera_pose_publisher",
    #     on_exit=launch.actions.Shutdown(),
    # )

    orbslam_node = Node(
        package="stretch_ros2_bridge",
        executable="orbslam3",
        name="orbslam3",
        on_exit=launch.actions.Shutdown(),
    )

    odometry_publisher_node = Node(
        package="stretch_ros2_bridge",
        executable="odom_tf_publisher",
        name="odom_tf_publisher",
        on_exit=launch.actions.Shutdown(),
    )

    state_estimator_node = Node(
        package="stretch_ros2_bridge",
        executable="state_estimator",
        name="state_estimator",
        output="screen",
        on_exit=launch.actions.Shutdown(),
    )

    # goto_controller_node = Node(
    #     package="stretch_ros2_bridge",
    #     executable="goto_controller",
    #     name="goto_controller",
    #     on_exit=launch.actions.Shutdown(),
    # )

    ld = LaunchDescription(
        [
            # start_robot_arg,
            stretch_driver_launch,
            realsense_launch,
            # camera_pose_publisher_node,
            state_estimator_node,
            # goto_controller_node,
            orbslam_node,
            odometry_publisher_node,
            declare_use_sim_time_argument,
        ]
    )

    ld.add_action(declare_use_sim_time_argument)

    return ld
