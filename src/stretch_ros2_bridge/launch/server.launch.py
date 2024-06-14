import os

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

    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("realsense2_camera"), "launch/rs_launch.py")
        ),
        launch_arguments={"align_depth.enable": "True"}.items(),
    )

    lidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("stretch_core"), "launch/rplidar.launch.py")
        )
    )

    use_sim_time = LaunchConfiguration("use_sim_time")
    slam_params_file = LaunchConfiguration("slam_params_file")

    start_async_slam_toolbox_node = Node(
        parameters=[slam_params_file, {"use_sim_time": use_sim_time}],
        package="slam_toolbox",
        executable="async_slam_toolbox_node",
        name="slam_toolbox",
        output="screen",
    )

    start_server = Node(
        package="stretch_ros2_bridge",
        executable="server",
        name="ros2_zmq_server",
        output="screen",
    )

    camera_pose_publisher_node = Node(
        package="stretch_ros2_bridge",
        executable="camera_pose_publisher",
        name="camera_pose_publisher",
    )

    odometry_publisher_node = Node(
        package="stretch_ros2_bridge",
        executable="odom_tf_publisher",
        name="odom_tf_publisher",
    )

    state_estimator_node = Node(
        package="stretch_ros2_bridge", executable="state_estimator", name="state_estimator"
    )

    goto_controller_node = Node(
        package="stretch_ros2_bridge", executable="goto_controller", name="goto_controller"
    )

    ld = LaunchDescription(
        [
            # start_robot_arg,
            stretch_driver_launch,
            realsense_launch,
            lidar_launch,
            camera_pose_publisher_node,
            state_estimator_node,
            goto_controller_node,
            odometry_publisher_node,
            declare_use_sim_time_argument,
            declare_slam_params_file_cmd,
            start_server,  # Add the ZMQ node
        ]
    )

    ld.add_action(declare_use_sim_time_argument)
    ld.add_action(declare_slam_params_file_cmd)
    ld.add_action(start_async_slam_toolbox_node)

    return ld
