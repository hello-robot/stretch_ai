# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import rclpy
from geometry_msgs.msg import Pose2D
from rclpy.node import Node
from slam_toolbox.srv import DeserializePoseGraph, SerializePoseGraph

from stretch.utils.memory import get_path_to_map


class MapSerializerDeserializer(Node):
    """
    A ROS2 node for serializing and deserializing maps using slam_toolbox services.
    """

    match_types = {
        "START_AT_FIRST_NODE": 1,
        "START_AT_GIVEN_POSE": 2,
        "LOCALIZE_AT_POSE": 3,
    }

    def __init__(self):
        super().__init__("map_serializer_deserializer")

        # Create clients for serialize and deserialize services
        self.serialize_client = self.create_client(
            SerializePoseGraph, "/slam_toolbox/serialize_map"
        )
        self.deserialize_client = self.create_client(
            DeserializePoseGraph, "/slam_toolbox/deserialize_map"
        )

        # Wait for services to become available
        while not self.serialize_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Serialize service not available, waiting...")

        while not self.deserialize_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Deserialize service not available, waiting...")

        # Initialize request objects
        self.serialize_request = SerializePoseGraph.Request()
        self.deserialize_request = DeserializePoseGraph.Request()

    def match_type_to_int(self, match_type: str):
        if match_type in self.match_types:
            return self.match_types[match_type]
        else:
            raise ValueError(f"match type {match_type} not supported")

    def serialize_map(self, name):
        """
        Serialize the current map and save it with the given name.

        Args:
            name (str): The name to save the map under.

        Returns:
            Future: A Future object representing the pending result of the service call.
        """
        self.serialize_request.filename = get_path_to_map(name)

        self.serialize_future = self.serialize_client.call_async(self.serialize_request)
        return self.serialize_future

    def deserialize_map(self, name, match_type: str = "START_AT_FIRST_NODE", initial_pose=None):
        """
        Deserialize a previously saved map.

        Args:
            name (str): The name of the map to deserialize.
            match_type (str): The method to use for initial localization.
                              Options: 'START_AT_FIRST_NODE', 'START_AT_GIVEN_POSE', 'LOCALIZE_AT_POSE'
            initial_pose (tuple): Initial pose as (x, y, theta) if match_type is 'START_AT_GIVEN_POSE'

        Returns:
            Future: A Future object representing the pending result of the service call.
        """
        self.deserialize_request.filename = get_path_to_map(name)
        self.deserialize_request.match_type = self.match_type_to_int(match_type)

        if match_type == "START_AT_GIVEN_POSE" or match_type == "LOCALIZE_AT_POSE":
            self.deserialize_request.initial_pose = Pose2D()
            self.deserialize_request.initial_pose.x = float(initial_pose[0])
            self.deserialize_request.initial_pose.y = float(initial_pose[1])
            self.deserialize_request.initial_pose.theta = float(initial_pose[2])

        self.deserialize_future = self.deserialize_client.call_async(self.deserialize_request)
        return self.deserialize_future


def main(args=None):
    """
    Main function to demonstrate map serialization and deserialization.
    """
    rclpy.init(args=args)

    map_handler = MapSerializerDeserializer()

    # Serialize map
    serialize_future = map_handler.serialize_map("test_map")

    while rclpy.ok():
        rclpy.spin_once(map_handler)
        if serialize_future.done():
            try:
                response = serialize_future.result()
                map_handler.get_logger().info("Map serialized successfully")
            except Exception as e:
                map_handler.get_logger().info(f"Serialization failed: {e}")
            break

    # Deserialize map
    deserialize_future = map_handler.deserialize_map(
        "test_map", match_type="LOCALIZE_AT_POSE", initial_pose=[0, 0, 0]
    )

    while rclpy.ok():
        rclpy.spin_once(map_handler)
        if deserialize_future.done():
            try:
                response = deserialize_future.result()
                map_handler.get_logger().info("Map deserialized successfully")
            except Exception as e:
                map_handler.get_logger().info(f"Deserialization failed: {e}")
            break

    # Clean up
    map_handler.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
