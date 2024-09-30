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
import threading

import numpy as np
import rclpy
import rclpy.time
from rclpy.time import Time
from sensor_msgs.msg import LaserScan


class RosLidar(object):
    """Simple wrapper node for a ROS lidar"""

    _max_dist = 100.0

    def __init__(self, ros_client, name: str = "/scan", verbose: bool = False):
        self.name = name
        self._points = None
        self.verbose = verbose
        self._lock = threading.Lock()
        self._t = Time()

        # print(f"robot client is: {type(robot_client)}")

        self._ros_client = ros_client
        self._subscriber = self._ros_client.create_subscription(
            LaserScan, self.name, self._lidar_scan_callback, 10
        )
        print("Done!!!!")

    def _lidar_scan_callback(self, scan_msg):
        # Get range and angle data from the scan message
        ranges = np.array(scan_msg.ranges)
        ranges[np.isnan(ranges)] = self._max_dist
        ranges[np.isinf(ranges)] = self._max_dist
        angles = np.linspace(scan_msg.angle_min, scan_msg.angle_max, len(ranges))

        # Convert polar coordinates (ranges, angles) to Cartesian coordinates (x, y)
        xs = ranges * np.cos(angles)
        ys = ranges * np.sin(angles)

        # Stack x and y coordinates to create a 2D NumPy array of points
        lidar_points = np.column_stack((xs, ys))

        # Now we have an array containing all the points from our lidar
        if self.verbose:
            print("[LIDAR] Lidar points:")
            print(lidar_points)
            print("[LIDAR] lidar time stamp:")
            print(scan_msg.header.stamp)

        with self._lock:
            # self._t = rclpy.time.Time()
            self._t = rclpy.time.Time.from_msg(scan_msg.header.stamp)
            self._points = lidar_points

    def get_time(self) -> rclpy.time.Time:
        """Get time image was received last"""
        return self._t

    def get(self) -> np.ndarray:
        """return the contents of the lidar (the last scan)"""
        with self._lock:
            return self._points

    def wait_for_scan(self) -> None:
        """Wait for image. Needs to be sort of slow, in order to make sure we give it time
        to update the image in the backend."""
        rate = self._ros_client.create_rate(5)
        while rclpy.ok():
            with self._lock:
                if self._points is not None:
                    break
            rate.sleep()


if __name__ == "__main__":
    import time

    from stretch_ros2_bridge.remote import StretchClient

    rclpy.init()

    client = StretchClient()

    print("Creating a lidar object..")
    time.sleep(5)
    lidar = RosLidar(ros_client=client.get_ros_client(), verbose=True)
    lidar.wait_for_scan()
    # rclpy.spin_once(client.get_ros_client())

    # print(lidar.get())
    # print(lidar.get_time())

    print("Ending main loop")
