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

    declare_slam_params_file_cmd = DeclareLaunchArgument(
        "slam_params_file",
        default_value=os.path.join(
            stretch_navigation_path, "config", "mapper_params_online_async.yaml"
        ),
        description="Full path to the ROS2 parameters file to use for the slam_toolbox node",
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
        "decimation_filter.enable": "True",
        "spatial_filter.enable": "True",
        "temporal_filter.enable": "True",
        "disparity_filter.enable": "True",
        "device_type": "d435i",
        "rgb_camera.profile": "1280x720x15",
        "depth_module.profile": "848x480x15",
    }
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("realsense2_camera"), "launch/rs_launch.py")
        ),
        launch_arguments=realsense_config.items(),
    )

    lidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("stretch_core"), "launch/rplidar.launch.py")
        )
    )

    # offline_mapping_launch = IncludeLaunchDescription(
    #    PythonLaunchDescriptionSource(
    #        # [get_package_share_directory("slam_toolbox"), "/launch/offline_launch.py"]
    #        [
    #            get_package_share_directory("slam_toolbox"),
    #            "/launch/online_async_launch.py",
    #        ]
    #    )
    # )

    use_sim_time = LaunchConfiguration("use_sim_time")
    slam_params_file = LaunchConfiguration("slam_params_file")

    declare_use_sim_time_argument = DeclareLaunchArgument(
        "use_sim_time", default_value="false", description="Use simulation/Gazebo clock"
    )
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

    state_estimator_node = Node(
        package="stretch_ros2_bridge",
        executable="state_estimator",
        name="state_estimator",
        output="screen",
        on_exit=launch.actions.Shutdown(),
    )

    goto_controller_node = Node(
        package="stretch_ros2_bridge",
        executable="goto_controller",
        name="goto_controller",
        on_exit=launch.actions.Shutdown(),
    )

    ld = LaunchDescription(
        [
            # start_robot_arg,
            stretch_driver_launch,
            # offline_mapping_launch,
            realsense_launch,
            lidar_launch,
            camera_pose_publisher_node,
            state_estimator_node,
            goto_controller_node,
            odometry_publisher_node,
            declare_use_sim_time_argument,
            declare_slam_params_file_cmd,
        ]
    )

    ld.add_action(declare_use_sim_time_argument)
    ld.add_action(declare_slam_params_file_cmd)
    ld.add_action(start_async_slam_toolbox_node)

    return ld
