#!/usr/bin/env python

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

import rclpy
import tf2_geometry_msgs
import tf2_ros
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node


class OdometryTFPublisher(Node):
    """Odometry tf pub class to resolve some issues"""

    def __init__(self):
        super().__init__("odometry_tf_publisher")
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self.odom_subscriber = self.create_subscription(Odometry, "/odom", self.odom_callback, 10)
        self.odom_subscriber

    def odom_callback(self, msg):
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = "odom"
        transform.child_frame_id = "base_link"
        transform.transform.translation.x = msg.pose.pose.position.x
        transform.transform.translation.y = msg.pose.pose.position.y
        transform.transform.translation.z = msg.pose.pose.position.z
        transform.transform.rotation.x = msg.pose.pose.orientation.x
        transform.transform.rotation.y = msg.pose.pose.orientation.y
        transform.transform.rotation.z = msg.pose.pose.orientation.z
        transform.transform.rotation.w = msg.pose.pose.orientation.w

        # tf_msg = tf2_geometry_msgs.TFMessage([transform])
        self.tf_broadcaster.sendTransform(transform)


def main(args=None):
    rclpy.init(args=args)
    node = OdometryTFPublisher()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
