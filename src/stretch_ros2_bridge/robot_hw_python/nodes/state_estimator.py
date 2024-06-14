#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import threading
from typing import Optional

import numpy as np
import rclpy
import sophuspy as sp
import tf2_ros
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, TransformStamped
from home_robot.motion.stretch import STRETCH_BASE_FRAME
from home_robot.utils.pose import to_matrix, transform_to_list
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from robot_hw_python.ros.utils import matrix_from_pose_msg, matrix_to_pose_msg
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

log = logging.getLogger(__name__)

SLAM_CUTOFF_HZ = 0.2


def cutoff_angle(duration, cutoff_freq):
    return 2 * np.pi * duration * cutoff_freq


class NavStateEstimator(Node):
    """Node that publishes transform between map and base_link"""

    def __init__(self, trust_slam: bool = False, use_history: bool = True):
        """Create nav state estimator.

        trust_slam: Just use the slam pose instead of odometry.
        use_history: Use previous filtered signals to compute current signal.
        """
        super().__init__("state_estimator")

        self.create_pubs_and_subs()

        self._trust_slam = trust_slam
        self._use_history = use_history

        # Create a lock to handle thread safety for pose updates
        self._slam_inject_lock = threading.Lock()

        self._filtered_pose = sp.SE3()
        self._slam_pose_sp = sp.SE3()
        self._slam_pose_prev = sp.SE3()
        self._t_odom_prev: Optional[Duration] = None
        self._pose_odom_prev = sp.SE3()

    def _publish_filtered_state(self, timestamp):
        if self._trust_slam:
            pose_msg = matrix_to_pose_msg(self._slam_pose_sp.matrix())
        else:
            pose_msg = matrix_to_pose_msg(self._filtered_pose.matrix())

        # Publish pose msg
        pose_out = PoseStamped()
        pose_out.header.stamp = timestamp
        pose_out.pose = pose_msg

        self._estimator_pub.publish(pose_out)

        # Publish to tf2
        t = TransformStamped()
        t.header.stamp = timestamp
        t.header.frame_id = self._world_frame_id
        t.child_frame_id = self._base_frame_id
        t.transform.translation.x = pose_msg.position.x
        t.transform.translation.y = pose_msg.position.y
        t.transform.translation.z = 0.0
        t.transform.rotation.x = pose_msg.orientation.x
        t.transform.rotation.y = pose_msg.orientation.y
        t.transform.rotation.z = pose_msg.orientation.z
        t.transform.rotation.w = pose_msg.orientation.w

        self._tf_broadcaster.sendTransform(t)

    def _filter_signals(
        self, slam_update: sp.SE3, odom_update: sp.SE3, t_interval: float
    ) -> sp.SE3:
        """
        The discrete high pass filter can be written as:
        ```
        coeff = dt / (RC + dt)  # RC = time constant
        output[t] = coeff * (output[t-1] + (input[t] - input[t-1]))
        ```

        The discrete low pass filter can be written as:
        ```
        coeff = dt / (RC + dt)  # RC = time constant
        output[t] = coeff * output[t-1] + (1 - coeff) * input[t]
        ```

        In the high pass filter case, we are injecting the output signal with the difference
        between measurements, while in the low pass filter case, we are injecting the output
        signal with the absolute value of the measurements.

        Slightly hand-wavy way of fusing the two filters to process the signals by simply adding
        together the injected signals of both slam + LPF and odom + HPF into the output pose:
        ```
        output_pose[t] = coeff * output_pose[t-1] \
            + (1 - coeff) * input_slam[t] \
            + coeff * (input_odom[t] - input_odom[t-1])
        ```
        which can be re-written as
        ```
        output_pose[t] = output_pose[t-1] \
            + (1 - coeff) * (input_slam[t] - output_pose[t-1]) \
            + coeff * (input_odom[t] - input_odom[t-1])
        ```

        References:
         - https://en.wikipedia.org/wiki/High-pass_filter#Algorithmic_implementation
         - https://en.wikipedia.org/wiki/Low-pass_filter#Simple_infinite_impulse_response_filter
        """
        # Compute mixing coefficient
        w = cutoff_angle(t_interval, SLAM_CUTOFF_HZ)
        coeff = 1 / (w + 1)

        # Compute pose differences
        with self._slam_inject_lock:
            if not self._use_history:
                pose_diff_slam = self._slam_pose_prev.inverse() * slam_update
                slam_pose = self._slam_pose_prev.copy()
            else:
                pose_diff_slam = self._filtered_pose.inverse() * slam_update
                slam_pose = self._filtered_pose

        pose_diff_odom = self._pose_odom_prev.inverse() * odom_update

        # Mix and inject signals
        pose_diff_log = coeff * pose_diff_odom.log() + (1 - coeff) * pose_diff_slam.log()
        return slam_pose * sp.SE3.exp(pose_diff_log)

    def _odom_callback(self, pose: Odometry):
        # self.get_clock().now() alternative for rospy.Time().now()
        t_curr = self.get_clock().now()

        # Compute injected signals into filtered pose
        pose_odom = sp.SE3(matrix_from_pose_msg(pose.pose.pose))

        # Update filtered pose
        if self._t_odom_prev is None:
            self._t_odom_prev = t_curr
        #
        t_interval_secs = (t_curr - self._t_odom_prev).nanoseconds * 1e-9

        self._filtered_pose = self._filter_signals(self._slam_pose_sp, pose_odom, t_interval_secs)
        self._publish_filtered_state(pose.header.stamp)

        # Update variables
        self._pose_odom_prev = pose_odom
        self._t_odom_prev = t_curr

    def _slam_pose_callback(self, pose: PoseWithCovarianceStamped) -> None:
        """Update slam pose for filtering"""
        self.get_logger().info(f"received pose {pose}")
        with self._slam_inject_lock:
            self._slam_pose_prev = self._slam_pose_sp
            self._slam_pose_sp = sp.SE3(matrix_from_pose_msg(pose.pose.pose))

    def get_pose(self):
        try:
            # Added transform_to_list function to handle change in return type of tf2 lookup_transform
            trans, rot = transform_to_list(
                self.tf_buffer.lookup_transform("map", STRETCH_BASE_FRAME, rclpy.time.Time())
            )
            matrix = to_matrix(trans, rot)

            with self._slam_inject_lock:
                self._slam_pose_prev = self._slam_pose_sp
                self._slam_pose_sp = sp.SE3(matrix)

        except TransformException as ex:
            self.get_logger().info(f"Could not tranform the base pose {ex}")

    def create_pubs_and_subs(self):
        # Create publishers and subscribers
        self._estimator_pub = self.create_publisher(
            PoseStamped, "/state_estimator/pose_filtered", 1
        )
        self._world_frame_id = "map"
        # TODO: if we need to debug this vs. the scan matcher
        # self._base_frame_id = "base_link_estimator"
        self._base_frame_id = "base_link"
        self._tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # This comes from hector_slam.
        # It's a transform from src_frame = 'base_link', target_frame = 'map'
        # The *inverse* is published by default from hector as the transform from map to base -
        # you can verify this with:
        #   rosrun tf tf_echo map base_link
        # Which will show the same output as this topic.
        # self.pose_subscriber = self.create_subscription(
        #     PoseWithCovarianceStamped,
        #     "/pose",
        #     self._slam_pose_callback,
        #     10,
        # )
        self.create_timer(1 / 10, self.get_pose)
        # This pose update comes from wheel odometry
        self.odom_subcriber = self.create_subscription(Odometry, "/odom", self._odom_callback, 1)

        # Run
        log.info("State Estimator launched.")


def main():
    rclpy.init()

    state_estimator = NavStateEstimator()
    rclpy.spin(state_estimator)

    state_estimator.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
