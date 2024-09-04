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
import threading
from typing import Optional

import numpy as np
import rclpy
import sophuspy as sp
import tf2_ros
from geometry_msgs.msg import Pose, PoseStamped, PoseWithCovarianceStamped, TransformStamped
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from std_msgs.msg import Bool
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from stretch.motion import STRETCH_BASE_FRAME
from stretch.utils.pose import to_matrix, transform_to_list
from stretch_ros2_bridge.ros.utils import matrix_from_pose_msg, matrix_to_pose_msg

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

        # Kalman filter parameters
        self.x = np.zeros(6)
        self.P = np.eye(6) * 1e-2
        self.Q = np.eye(6) * 1e-5
        self.R = np.eye(6) * 1e-3
        self.H = np.eye(6)

        dt = 0.1  # TODO: get from odom message
        self.F = np.eye(8)

        self.slam = None

        self.measurement1 = None  # wheel odometry
        self.measurement2 = None  # 2D SLAM (Hector)
        self.measurement3 = None  # vio (ORB-SLAM3)

        # ORB-SLAM3 tracking state
        self.orb_slam3_tracking_ok = True

    def predict_kalman(self):
        """
        Predict the state of the Kalman filter.
        Propagates the state vector through the state transition matrix F.
        Covariance matrix is updated using the process noise covariance matrix Q.

        x = F @ x              # This step predicts the next state
        P = F @ P @ F.T + Q    # This step predicts the next covariance
        """
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update_kalman(self, z):
        """
        Update the state of the Kalman filter.

        Parameters:
        z (np.array): Measurement vector.

        y = z - H @ x         # This step computes the innovation or measurement residual
        S = H @ P @ H.T + R   # This step computes the innovation covariance
        K = P @ H.T @ inv(S)  # This step computes the Kalman gain
        x = x + K @ y         # This step updates the state estimate (posterior)
        P = (I - K @ H) @ P   # This step updates the covariance estimate (posterior)
        """
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P

    def fuse_measurements(self, wheel_measurement, slam_measurement, vio_measurement=None):
        """
        Fuse the measurements from wheel odometry, 2D SLAM, and VIO.

        Parameters:
        wheel_measurement (list): Wheel odometry measurement.
        slam_measurement (list): 2D SLAM measurement.
        vio_measurement (list): VIO measurement.
        """
        if vio_measurement is not None:
            combined_measurement = (
                np.array(wheel_measurement) + np.array(slam_measurement) + np.array(vio_measurement)
            ) / 3
        else:
            combined_measurement = (np.array(wheel_measurement) + np.array(slam_measurement)) / 2
        self.update_kalman(combined_measurement)
        self.publish_kf_state()

        self.measurement1 = None
        self.measurement2 = None
        self.measurement3 = None

    def publish_kf_state(self):
        """
        Publish the state of the Kalman filter.
        """
        pose_msg = Pose()
        pose_msg.position.x = self.x[0]
        pose_msg.position.y = self.x[1]
        pose_msg.position.z = self.x[2]
        pose_msg.orientation.x = self.x[3]
        pose_msg.orientation.y = self.x[4]
        pose_msg.orientation.z = self.x[5]
        pose_msg.orientation.w = 1.0

        pose_out = PoseStamped()
        pose_out.header.stamp = self.get_clock().now().to_msg()
        pose_out.header.frame_id = "base_link"
        pose_out.pose = pose_msg

        self._estimator_kf_pub.publish(pose_out)

    def _publish_filtered_state(self, timestamp):
        """
        Publish the filtered state of the robot.

        Parameters:
        timestamp (rclpy.time.Time): Timestamp of the message.
        """
        if self._trust_slam:
            pose_msg = matrix_to_pose_msg(self._slam_pose_sp.matrix())
        else:
            pose_msg = matrix_to_pose_msg(self._filtered_pose.matrix())

        # Publish pose msg
        pose_out = PoseStamped()
        pose_out.header.stamp = timestamp
        pose_out.header.frame_id = "base_link"
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

    def _vio_odom_callback(self, pose: PoseStamped):
        """
        Callback for VIO odometry.

        Parameters:
        pose (PoseStamped): VIO odometry pose.
        """
        # self.get_clock().now() alternative for rospy.Time().now()
        t_curr = self.get_clock().now()

        # Compute injected signals into filtered pose
        pose_odom = sp.SE3(matrix_from_pose_msg(pose.pose))

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

        self.measurement3 = [
            pose.pose.position.x,
            pose.pose.position.y,
            pose.pose.position.z,
            pose.pose.orientation.x,
            pose.pose.orientation.y,
            pose.pose.orientation.z,
        ]

        if self.measurement2 is not None and self.measurement1 is not None:
            if self.orb_slam3_tracking_ok:
                self.fuse_measurements(self.measurement1, self.measurement2, self.measurement3)
            else:
                self.fuse_measurements(self.measurement1, self.measurement2)

    def _wheel_odom_callback(self, pose: Odometry):
        """
        Callback for wheel odometry.

        Parameters:
        pose (Odometry): Wheel odometry pose.
        """
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

        self.measurement1 = [
            pose.pose.pose.position.x,
            pose.pose.pose.position.y,
            pose.pose.pose.position.z,
            pose.pose.pose.orientation.x,
            pose.pose.pose.orientation.y,
            pose.pose.pose.orientation.z,
        ]

        if self.measurement2 is not None and self.measurement3 is not None:
            if self.orb_slam3_tracking_ok:
                self.fuse_measurements(self.measurement1, self.measurement2, self.measurement3)
            else:
                self.fuse_measurements(self.measurement1, self.measurement2)

    def _slam_pose_callback(self, pose: PoseWithCovarianceStamped) -> None:
        """Update slam pose for filtering

        Parameters:
        pose (PoseWithCovarianceStamped): Slam pose
        """
        self.get_logger().info(f"received pose {pose}")
        with self._slam_inject_lock:
            self._slam_pose_prev = self._slam_pose_sp
            self._slam_pose_sp = sp.SE3(matrix_from_pose_msg(pose.pose.pose))

        self.measurement2 = [
            pose.pose.pose.position.x,
            pose.pose.pose.position.y,
            pose.pose.pose.position.z,
            pose.pose.pose.orientation.x,
            pose.pose.pose.orientation.y,
            pose.pose.pose.orientation.z,
        ]

        if self.measurement1 is not None and self.measurement3 is not None:
            if self.orb_slam3_tracking_ok:
                self.fuse_measurements(self.measurement1, self.measurement2, self.measurement3)
            else:
                self.fuse_measurements(self.measurement1, self.measurement2)

    def _vio_status_callback(self, status: Bool):
        """
        Callback for VIO status.

        Parameters:
        status (Bool): VIO status.
        True if ORB_SLAM3 is tracking landmarks well, False otherwise.
        """
        self.orb_slam3_tracking_ok = status.data

    def get_pose(self):
        """
        Get the pose of the robot in the map frame and update the filtered pose.
        """
        try:
            # Added transform_to_list function to handle change in return type of tf2 lookup_transform
            trans, rot = transform_to_list(
                self.tf_buffer.lookup_transform("map", STRETCH_BASE_FRAME, rclpy.time.Time())
            )
            matrix = to_matrix(trans, rot)

            with self._slam_inject_lock:
                self._slam_pose_prev = self._slam_pose_sp
                self._slam_pose_sp = sp.SE3(matrix)

            self.measurement2 = [trans[0], trans[1], trans[2], rot[0], rot[1], rot[2]]

            if self.measurement1 is not None and self.measurement3 is not None:
                if self.orb_slam3_tracking_ok:
                    self.fuse_measurements(self.measurement1, self.measurement2, self.measurement3)
                else:
                    self.fuse_measurements(self.measurement1, self.measurement2)

        except TransformException as ex:
            self.get_logger().info(f"Could not transform the base pose {ex}")

    def create_pubs_and_subs(self):
        """
        Create publishers and subscribers.
        """
        # Create publishers and subscribers
        self._estimator_pub = self.create_publisher(
            PoseStamped, "/state_estimator/pose_filtered", 1
        )
        self._estimator_kf_pub = self.create_publisher(PoseStamped, "/state_estimator/pose_kf", 1)
        self._world_frame_id = "map"
        # TODO: if we need to debug this vs. the scan matcher
        # self._base_frame_id = "base_link_estimator"
        self._base_frame_id = "base_link_estimator"
        self._tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # This comes from hector_slam.
        # It's a transform from src_frame = 'base_link', target_frame = 'map'
        # The *inverse* is published by default from hector as the transform from map to base -
        # you can verify this with:
        #   rosrun tf tf_echo map base_link
        # Which will show the same output as this topic.
        self.pose_subscriber = self.create_subscription(
            PoseWithCovarianceStamped,
            "/pose",
            self._slam_pose_callback,
            10,
        )
        self.create_timer(1 / 10, self.get_pose)

        # This pose update comes from wheel odometry
        self.wheel_odom_subcriber = self.create_subscription(
            Odometry, "/odom", self._wheel_odom_callback, 1
        )

        # VIO odometry
        self.vio_odom_subcriber = self.create_subscription(
            PoseStamped, "/orb_slam3/pose", self._vio_odom_callback, 1
        )

        # VIO status
        self.vio_status_subcriber = self.create_subscription(
            Bool, "/orb_slam3/tracking_status", self._vio_status_callback, 1
        )

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
