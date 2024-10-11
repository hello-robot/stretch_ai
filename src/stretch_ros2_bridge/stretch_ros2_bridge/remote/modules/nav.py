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
import time
from typing import List

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from std_srvs.srv import SetBool, Trigger

from stretch.motion.robot import RobotModel
from stretch.utils.geometry import angle_difference, sophus2xyt, xyt2sophus, xyt_base_to_global
from stretch_ros2_bridge.constants import T_LOC_STABILIZE
from stretch_ros2_bridge.ros.utils import matrix_to_pose_msg

from .abstract import AbstractControlModule, enforce_enabled


class StretchNavigationClient(AbstractControlModule):
    block_spin_rate = 10

    def __init__(self, ros_client, robot_model: RobotModel):
        super().__init__()

        self._ros_client = ros_client
        self._robot_model = robot_model
        self._wait_for_pose()

    # Enable / disable

    def _enable_hook(self) -> bool:
        """Called when interface is enabled."""
        result = self._ros_client.nav_mode_service.call(Trigger.Request())
        self._ros_client.get_logger().info(result.message)
        return result.success

    def _disable_hook(self) -> bool:
        """Called when interface is disabled."""
        result = self._ros_client.goto_off_service.call(Trigger.Request())

        rate = self._ros_client.create_rate(1 / T_LOC_STABILIZE)
        rate.sleep()  # wait for robot movement to stop
        return result.success

    # Interface methods

    def get_base_pose(self, matrix=False) -> np.ndarray:
        """get the latest base pose from sensors"""
        if not matrix:
            return sophus2xyt(self._ros_client.se3_base_filtered)
        else:
            return self._ros_client.se3_base_filtered.matrix()

    def at_goal(self) -> bool:
        """Returns true if the agent is currently at its goal location"""
        # self._ros_client.get_logger().info(
        #     f"time diff {self._ros_client.get_clock().now()} - {self._ros_client._goal_reset_t}"
        # )
        # self._ros_client.get_logger().info(f"goal rewset time now {self._ros_client._goal_reset_t}")
        if (
            self._ros_client._goal_reset_t is not None
            and (self._ros_client.get_clock().now() - self._ros_client._goal_reset_t).nanoseconds
            * 1e-9
            > self._ros_client.msg_delay_t
        ):
            return self._ros_client.at_goal
        else:
            return False

    def wait_for_waypoint(
        self,
        xyt: np.ndarray,
        rate: int = 10,
        pos_err_threshold: float = 0.2,
        rot_err_threshold: float = 0.75,
        verbose: bool = True,
        timeout: float = 10.0,
    ) -> bool:
        """Wait until the robot has reached a configuration... but only roughly. Used for trajectory execution.

        Parameters:
            xyt: se(2) base pose in world coordinates to go to
            rate: rate at which we should check to see if done
            pos_err_threshold: how far robot can be for this waypoint
            verbose: prints extra info out
            timeout: aborts at this point

        Returns:
            success: did we reach waypoint in time"""
        _rate = self._ros_client.create_rate(rate)
        xy = xyt[:2]
        if verbose:
            print("Waiting for", xyt, "threshold =", pos_err_threshold)
        # Save start time for exiting trajectory loop
        t0 = self._ros_client.get_clock().now()
        while rclpy.ok():
            # Loop until we get there (or time out)
            curr = self.get_base_pose()
            pos_err = np.linalg.norm(xy - curr[:2])
            rot_err = np.abs(angle_difference(curr[-1], xyt[2]))
            if verbose:
                print(f"- {curr=} target {xyt=} {pos_err=} {rot_err=}")
            if pos_err < pos_err_threshold and rot_err < rot_err_threshold:
                # We reached the goal position
                return True
            t1 = self._ros_client.get_clock().now()
            if ((t1 - t0).nanoseconds * 1e-9) > timeout:
                self._ros_client.get_logger().info("Could not reach goal in time: " + str(xyt))
                return False
            _rate.sleep()
        return False

    @enforce_enabled
    def execute_trajectory(
        self,
        trajectory: List[np.ndarray],
        pos_err_threshold: float = 0.2,
        rot_err_threshold: float = 0.75,
        spin_rate: int = 10,
        verbose: bool = False,
        per_waypoint_timeout: float = 10.0,
        relative: bool = False,
    ):
        """Execute a multi-step trajectory; this is always blocking since it waits to reach each one in turn."""
        for i, pt in enumerate(trajectory):
            assert (
                len(pt) == 3 or len(pt) == 2
            ), "base trajectory needs to be 2-3 dimensions: x, y, and (optionally) theta"
            just_xy = len(pt) == 2
            self.navigate_to(pt, relative, position_only=just_xy, blocking=False)
            self.wait_for_waypoint(
                pt,
                pos_err_threshold=pos_err_threshold,
                rot_err_threshold=rot_err_threshold,
                rate=spin_rate,
                verbose=verbose,
                timeout=per_waypoint_timeout,
            )
        self.navigate_to(pt, blocking=True)

    @enforce_enabled
    def set_velocity(self, v, w):
        """
        Directly sets the linear and angular velocity of robot base.
        """
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)

        self._ros_client.goto_off_service.call(Trigger.Request())
        self._ros_client.velocity_pub.publish(msg)

    @enforce_enabled
    def navigate_to(
        self,
        xyt: List[float],
        relative: bool = False,
        position_only: bool = False,
        avoid_obstacles: bool = False,
        blocking: bool = True,
    ):
        """
        Cannot be used in manipulation mode.
        """
        # Parse inputs
        assert len(xyt) == 3, "Input goal location must be of length 3."

        if avoid_obstacles:
            raise NotImplementedError("Obstacle avoidance unavailable.")

        self._ros_client.get_logger().info("Setting yaw service")
        # Set yaw tracking
        self._ros_client.set_yaw_service.call(SetBool.Request(data=(not position_only)))
        self._ros_client.get_logger().info("Yaw service set")

        # Compute absolute goal
        if relative:
            xyt_base = sophus2xyt(self._ros_client.se3_base_filtered)
            xyt_goal = xyt_base_to_global(xyt, xyt_base)
        else:
            xyt_goal = xyt
        self._ros_client.get_logger().info(f"Sending XYT Goal {xyt_goal=}")

        # Clear self.at_goal
        self._ros_client.at_goal = False
        self._ros_client.goal_reset_t = None

        # Set goal
        goal_matrix = xyt2sophus(xyt_goal).matrix()
        self._ros_client.goal_visualizer(goal_matrix)
        msg = matrix_to_pose_msg(goal_matrix)

        # self._ros_client.get_logger().info("Setting goto service call")
        self._ros_client.goto_on_service.call(Trigger.Request())
        # self._ros_client.get_logger().info("Goto service call set")

        self._ros_client.goal_pub.publish(msg)

        # self._ros_client.get_logger().info("Waiting for completion")
        self._register_wait(self._wait_for_goal_reached)
        if blocking:
            self.wait()

    @enforce_enabled
    def home(self):
        self.navigate_to([0.0, 0.0, 0.0], blocking=True)

    # Helper methods

    def _wait_for_pose(self):
        """wait until we have an accurate pose estimate"""
        rate = self._ros_client.create_rate(10)
        while rclpy.ok():
            if self._ros_client.se3_base_filtered is not None:
                break
            rate.sleep()

    def _wait_for_goal_reached(self, verbose: bool = False):
        """Wait until goal is reached"""
        # rate = self._ros_client.create_rate(1 / self._ros_client.msg_delay_t)
        # rate.sleep()
        time.sleep(self._ros_client.msg_delay_t)

        rate = self._ros_client.create_rate(self.block_spin_rate)
        t0 = self._ros_client.get_clock().now()
        while rclpy.ok():
            t1 = self._ros_client.get_clock().now()
            if verbose:
                print(
                    "...waited for controller",
                    (t1 - t0).to_sec(),
                    "is at goal =",
                    self.at_goal(),
                )
            # Verify that we are at goal and perception is synchronized with pose
            if self.at_goal() and self._ros_client.recent_depth_image(self._ros_client.msg_delay_t):
                break
            else:
                rate.sleep()
        # TODO: this should be unnecessary
        # TODO: add this back in if we are having trouble building maps
        # Make sure that depth and position are synchronized
        # rospy.sleep(self.msg_delay_t * 5)
