# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

from typing import Optional

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.exceptions import ROSInterruptException
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_srvs.srv import Trigger


class StreamingActivator:
    """Activate streaming node."""

    def __init__(self, node: Optional[Node] = None):
        if node is None:
            self._node = Node("streaming_activator")
        else:
            self._node = node
        self.cb_group = ReentrantCallbackGroup()
        self.client = self._node.create_client(
            Trigger, "/activate_streaming_position", callback_group=self.cb_group
        )

    def activate_streaming(self) -> bool:
        print("Activating streaming mode in the ROS2 driver...")
        if not self.client.wait_for_service(timeout_sec=10.0):
            self._node.get_logger().error("Service /activate_streaming_position not available")
            return False

        request = Trigger.Request()
        response = self.client.call(request)
        result = response.success
        if not result:
            self._node.get_logger().error("Could not activate streaming service!")
        return result


class StreamingController:
    """Version of the activator class designed to both activate and deactivate."""

    def __init__(self, node: Optional[Node] = None):
        if node is None:
            self._node = Node("streaming_controller")
        else:
            self._node = node
        self.cb_group = ReentrantCallbackGroup()
        self.activate_client = self._node.create_client(
            Trigger, "/activate_streaming_position", callback_group=self.cb_group
        )
        self.deactivate_client = self._node.create_client(
            Trigger, "/deactivate_streaming_position", callback_group=self.cb_group
        )

    def call_service(self, client, service_name):
        if not client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error(f"Service {service_name} not available")
            return False

        request = Trigger.Request()
        future = client.call_async(request)

        try:
            rclpy.spin_until_future_complete(self._node, future, timeout_sec=10.0)
        except ROSInterruptException:
            self.get_logger().error(f"Interrupted while waiting for the service {service_name}")
            return False

        if future.result() is None:
            self.get_logger().error(f"Service call to {service_name} failed")
            return False

        return future.result().success

    def activate_streaming(self):
        return self.call_service(self.activate_client, "/activate_streaming_position")

    def deactivate_streaming(self):
        return self.call_service(self.deactivate_client, "/deactivate_streaming_position")


def main(args=None):
    rclpy.init(args=args)
    activator = StreamingActivator()
    executor = MultiThreadedExecutor()

    result = activator.activate_streaming()
    if result:
        activator.get_logger().info("Streaming activated successfully")
    else:
        activator.get_logger().error("Failed to activate streaming")

    activator.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
