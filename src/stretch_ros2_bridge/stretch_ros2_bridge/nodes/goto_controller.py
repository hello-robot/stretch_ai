#!/usr/bin/env python
# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
from typing import Optional

import numpy as np
import rclpy
import sophuspy as sp
from geometry_msgs.msg import Pose, PoseStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.clock import ClockType
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import Bool, Float32
from std_srvs.srv import SetBool, Trigger

from stretch.motion.control.goto_controller import GotoVelocityController
from stretch.utils.config import get_control_config
from stretch.utils.geometry import sophus2xyt
from stretch_ros2_bridge.ros.utils import matrix_from_pose_msg
from stretch_ros2_bridge.ros.visualizer import Visualizer

log = logging.getLogger(__name__)

CONTROL_HZ = 20
VEL_THRESHOlD = 0.001
RVEL_THRESHOLD = 0.005
DEBUG_CONTROL_LOOP = True


class GotoVelocityControllerNode(Node):
    """
    Self-contained controller module for moving a diff drive robot to a target goal.
    Target goal is update-able at any given instant.
    """

    def __init__(
        self,
        hz: float,
        odom_only_feedback: bool = False,
        config_name: str = "noplan_velocity_hw",
    ):
        super().__init__("goto_controller")

        self.hz = hz
        self.odom_only = odom_only_feedback

        # How long should the controller report done before we're actually confident that we're done?
        self.done_t = Duration(seconds=0.1)

        # Control module
        controller_cfg = get_control_config(config_name)
        self.controller = GotoVelocityController(controller_cfg)
        # Update the velocity and acceleration configs from the file
        self.controller.update_velocity_profile(
            controller_cfg.v_max,
            controller_cfg.w_max,
            controller_cfg.acc_lin,
            controller_cfg.acc_ang,
        )

        # Initialize
        self.vel_odom: Optional[np.ndarray] = None
        self.xyt_filtered: Optional[np.ndarray] = None
        self.xyt_goal: Optional[np.ndarray] = None

        self.active = False
        self.is_done = True
        self.controller_finished = True
        self.done_since = Time(clock_type=ClockType.ROS_TIME)
        self.track_yaw = True
        self.goal_set_t = Time(clock_type=ClockType.ROS_TIME)

        # Visualizations
        self.goal_visualizer = Visualizer("goto_controller/goal_abs")

        self.create_pubs_subs_timers()

    def _set_v_max(self, msg):
        self.controller.update_velocity_profile(v_max=msg.data)

    def _set_w_max(self, msg):
        self.controller.update_velocity_profile(w_max=msg.data)

    def _set_acc_lin(self, msg):
        self.controller.update_velocity_profile(acc_lin=msg.data)

    def _set_acc_ang(self, msg):
        self.controller.update_velocity_profile(acc_ang=msg.data)

    def _pose_update_callback(self, msg: PoseStamped):
        pose_sp = sp.SE3(matrix_from_pose_msg(msg.pose))
        self.xyt_filtered = sophus2xyt(pose_sp)
        if not self.odom_only:
            self.controller.update_pose_feedback(self.xyt_filtered)

    def _odom_update_callback(self, msg: Odometry):
        pose_sp = sp.SE3(matrix_from_pose_msg(msg.pose.pose))
        self.vel_odom = np.array([msg.twist.twist.linear.x, msg.twist.twist.angular.z])
        if self.odom_only:
            self.controller.update_pose_feedback(sophus2xyt(pose_sp))

    def _goal_update_callback(self, msg: Pose):
        pose_sp = sp.SE3(matrix_from_pose_msg(msg))

        """
        if self.odom_only:
            # Project absolute goal from current odometry reading
            pose_delta = xyt2sophus(self.xyt_loc_odom).inverse() * pose_sp
            pose_goal = xyt2sophus(self.xyt_loc_odom) * pose_delta
        else:
            # Assign absolute goal directly
            pose_goal = pose_sp
        """

        if self.active:
            pose_goal = pose_sp

            self.controller.update_goal(sophus2xyt(pose_goal))
            self.xyt_goal = self.controller.xyt_goal

            self.is_done = False
            self.goal_set_t = self.get_clock().now()
            self.controller_finished = False

            # Visualize
            # self.goal_visualizer(pose_goal.matrix())

        # Do not update goal if controller is not active (prevents _enable_service to suddenly start moving the robot)
        else:
            self.get_logger().warn("Received a goal while NOT active. Goal is not updated.")

    def _enable_service(self, request, response):
        """activates the controller and acks activation request"""
        self.xyt_goal = None
        self.active = True

        response.success = True
        response.message = "Goto controller is now RUNNING"
        return response

    def _disable_service(self, request, response):
        """disables the controller and acks deactivation request"""
        self.active = False
        self.xyt_goal = None

        response.success = True
        response.message = "Goto controller is now STOPPED"
        return response

    def _set_yaw_tracking_service(self, request: SetBool, response):
        track_yaw = request.data

        self.controller.set_yaw_tracking(track_yaw)

        status_str = "ON" if self.track_yaw else "OFF"

        response.success = True
        response.message = f"Yaw tracking is now {status_str}"
        return response

    def _set_velocity(self, v_m, w_r):
        cmd = Twist()
        cmd.linear.x = v_m
        cmd.angular.z = w_r
        self.vel_command_pub.publish(cmd)

    def control_loop_callback(self):
        """Actual controller timer callback"""

        if self.active and self.xyt_goal is not None:
            # Compute control
            self.is_done = False
            v_cmd, w_cmd = self.controller.compute_control()
            done = self.controller.is_done()

            # self.get_logger().info(f"veclocities {v_cmd} and {w_cmd}")
            # Compute timeout
            time_since_goal_set = (self.get_clock().now() - self.goal_set_t).nanoseconds * 1e-9
            if self.controller.timeout(time_since_goal_set):
                done = True
                v_cmd, w_cmd = 0, 0

            # Check if actually done (velocity = 0)
            if done and self.vel_odom is not None:
                if self.vel_odom[0] < VEL_THRESHOlD and self.vel_odom[1] < RVEL_THRESHOLD:
                    if not self.controller_finished:
                        self.controller_finished = True
                        self.done_since = self.get_clock().now()
                    elif (
                        self.controller_finished
                        and (self.get_clock().now() - self.done_since) > self.done_t
                    ):
                        self.is_done = True
                else:
                    self.controller_finished = False
                    self.done_since = Time(clock_type=ClockType.ROS_TIME)

            # self.get_logger().info(
            #     f"done = {done} cmd vel = {v_cmd, w_cmd} odom vel = {self.vel_odom} controller done = {self.controller_finished} is done ={self.is_done}"
            # )
            # Command robot
            self._set_velocity(v_cmd * 1.0, w_cmd * 1.0)
            self.at_goal_pub.publish(Bool(data=self.is_done))

            if self.is_done:
                self.active = False
                self.xyt_goal = None

    def create_pubs_subs_timers(self):
        """Publishers and Subscribers"""

        self.vel_command_pub = self.create_publisher(Twist, "stretch/cmd_vel", 1)
        self.at_goal_pub = self.create_publisher(Bool, "goto_controller/at_goal", 1)

        self.create_subscription(
            PoseStamped, "state_estimator/pose_filtered", self._pose_update_callback, 1
        )
        self.create_subscription(Odometry, "odom", self._odom_update_callback, 1)
        self.create_subscription(Pose, "goto_controller/goal", self._goal_update_callback, 1)

        # Create individual subscribers
        self.create_subscription(Float32, "goto_controller/v_max", self._set_v_max, 1)
        self.create_subscription(Float32, "goto_controller/w_max", self._set_w_max, 1)
        self.create_subscription(Float32, "goto_controller/acc_lin", self._set_acc_lin, 1)
        self.create_subscription(Float32, "goto_controller/acc_ang", self._set_acc_ang, 1)

        self.create_service(Trigger, "goto_controller/enable", self._enable_service)
        self.create_service(Trigger, "goto_controller/disable", self._disable_service)
        self.create_service(
            SetBool, "goto_controller/set_yaw_tracking", self._set_yaw_tracking_service
        )

        # Run controller
        self.create_timer(1 / self.hz, self.control_loop_callback)
        self.get_logger().info("Goto Controller launched.")


def main():
    rclpy.init()

    velocity_controller = GotoVelocityControllerNode(CONTROL_HZ)
    rclpy.spin(velocity_controller)

    velocity_controller.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
