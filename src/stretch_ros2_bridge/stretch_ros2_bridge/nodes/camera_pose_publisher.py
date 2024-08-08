# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import numpy as np
import rclpy
import trimesh.transformations as tra
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node

# import tf
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from stretch.motion import STRETCH_CAMERA_FRAME
from stretch.utils.pose import to_matrix, transform_to_list
from stretch_ros2_bridge.ros.utils import matrix_to_pose_msg


class CameraPosePublisher(Node):
    """Node that publishes camera pose transform [map -> camer_frame]"""

    def __init__(self, topic_name: str = "camera_pose"):
        super().__init__("camera_pose_publisher")

        self._pub = self.create_publisher(PoseStamped, topic_name, 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self._seq = 0

        timer_freq = 10
        self.timer = self.create_timer(1 / timer_freq, self.timer_callback)

    def timer_callback(self):
        """Transform Callback"""

        try:
            # Added transform_to_list function to handle change in return type of tf2 lookup_transform
            trans, rot = transform_to_list(
                self.tf_buffer.lookup_transform("map", STRETCH_CAMERA_FRAME, rclpy.time.Time())
            )
            matrix = to_matrix(trans, rot)

            # We rotate by 90 degrees from the frame of realsense hardware since we are also rotating images to be upright
            matrix_rotated = matrix @ tra.euler_matrix(0, 0, -np.pi / 2)

            msg = PoseStamped(pose=matrix_to_pose_msg(matrix_rotated))
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = str(self._seq)
            self._pub.publish(msg)
            self._seq += 1
        except TransformException as ex:
            self.get_logger().info(f"Could not transform the camera pose {ex}")


def main():
    """Init and Spin the Node"""

    rclpy.init()

    camera_pose_publisher = CameraPosePublisher()
    rclpy.spin(camera_pose_publisher)

    camera_pose_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
