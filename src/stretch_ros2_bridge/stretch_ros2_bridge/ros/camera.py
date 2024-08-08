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
from collections import deque

import numpy as np
import rclpy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image

from stretch.utils.image import Camera
from stretch_ros2_bridge.ros.msg_numpy import image_to_numpy


class RosCamera(Camera):
    """compute camera parameters from ROS instead"""

    def __init__(
        self,
        ros_client,
        name: str = "/camera/color",
        verbose: bool = True,
        rotations: int = 0,
        buffer_size: int = None,
        image_ext: str = "/image_raw",
    ):
        """
        Args:
            name: Image topic name
            verbose: Whether or not to print out camera info
            rotations: Number of counterclockwise rotations for the output image array
            buffer_size: Size of buffer for initialization and filtering
        """

        self._ros_client = ros_client
        self.name = name
        self.rotations = rotations

        # Initialize
        self._img = None
        self._t = Time()
        self._lock = threading.Lock()
        print(name)
        self._camera_info_topic = name + "/camera_info"

        if verbose:
            print("Waiting for camera info on", self._camera_info_topic + "...")

        self._info_sub = self._ros_client.create_subscription(
            CameraInfo, self._camera_info_topic, self.cam_info_callback, 100
        )
        cam_info = self.wait_for_camera_info()
        print("Camera info:", cam_info)

        # Buffer
        self.buffer_size = buffer_size
        if self.buffer_size is not None:
            # create buffer
            self._buffer: deque = deque()

        self.height = cam_info.height
        self.width = cam_info.width
        self.pos, self.orn, self.pose_matrix = None, None, None

        # Get camera information
        self.distortion_model = cam_info.distortion_model
        self.D = np.array(cam_info.d)  # Distortion parameters
        self.K = np.array(cam_info.k).reshape(3, 3)
        self.R = np.array(cam_info.r).reshape(3, 3)  # Rectification matrix
        self.P = np.array(cam_info.p).reshape(3, 4)  # Projection/camera matrix

        if self.rotations % 2 != 0:
            self.K[0, 0], self.K[1, 1] = self.K[1, 1], self.K[0, 0]
            self.K[0, 2], self.K[1, 2] = self.K[1, 2], self.K[0, 2]
            self.P[0, 0], self.P[1, 1] = self.P[1, 1], self.P[0, 0]
            self.P[0, 2], self.P[1, 2] = self.P[1, 2], self.P[0, 2]
            self.height = cam_info.width
            self.width = cam_info.height

        self.fx = self.K[0, 0]
        self.fy = self.K[1, 1]
        self.px = self.K[0, 2]
        self.py = self.K[1, 2]

        self.near_val = 0.1
        self.far_val = 5.0
        if verbose:
            print()
            print("---------------")
            print("Created camera with info:")
            print(cam_info)
            print("---------------")
        self.frame_id = cam_info.header.frame_id

        # Get the
        if verbose:
            print("... this is the", self.name.split("/")[-1], "camera.")
        self.topic_name = name + image_ext
        self._sub = self._ros_client.create_subscription(Image, self.topic_name, self._cb, 1)

    def cam_info_callback(self, msg):
        """Camera Info callback"""
        self.camera_info = msg

    def wait_for_camera_info(self) -> CameraInfo:
        """Wait until you get the camera info"""

        self.camera_info = None
        rate = self._ros_client.create_rate(100)
        while self.camera_info is None:
            rate.sleep()
        return self.camera_info

    def _cb(self, msg):
        """capture the latest image and save it"""
        with self._lock:
            img = image_to_numpy(msg)

            # Preprocess encoding
            if msg.encoding == "16UC1":
                # depth support goes here
                # Convert the image to metric (meters)
                img = img / 1000.0
            elif msg.encoding == "rgb8":
                # color support - do nothing
                pass

            # Image orientation
            self._img = np.rot90(img, k=self.rotations)

            # Add to buffer
            self._t = msg.header.stamp
            if self.buffer_size is not None:
                self._add_to_buffer(img)

    def _add_to_buffer(self, img):
        """add to buffer and remove old image if buffer size exceeded"""
        self._buffer.append(img)
        if len(self._buffer) > self.buffer_size:
            self._buffer.popleft()

    def valid_mask(self, depth):
        """return only valid pixels"""
        depth = depth.reshape(-1)
        return np.bitwise_and(depth > self.near_val, depth < self.far_val)

    def valid_pc(self, xyz, rgb, depth):
        mask = self.valid_mask(depth)
        xyz = xyz.reshape(-1, 3)[mask]
        rgb = rgb.reshape(-1, 3)[mask]
        return xyz, rgb

    def get_time(self):
        """Get time image was received last"""
        return self._t

    def wait_for_image(self) -> None:
        """Wait for image. Needs to be sort of slow, in order to make sure we give it time
        to update the image in the backend."""
        # rospy.sleep(0.2)
        rate = self._ros_client.create_rate(5)
        while rclpy.ok():
            with self._lock:
                if self.buffer_size is None:
                    if self._img is not None:
                        break
                else:
                    # Wait until we have a full buffer
                    if len(self._buffer) >= self.buffer_size:
                        break
            rate.sleep()

    def get(self, device=None):
        """return the current image associated with this camera"""
        with self._lock:
            if self._img is None:
                return None
            else:
                # Return a copy
                img = self._img.copy()

        if device is not None:
            # If a device is specified, assume we want to move to pytorch
            import torch

            img = torch.FloatTensor(img).to(device)

        return img

    def get_filtered(self, std_threshold=0.005, device=None):
        """get image from buffer; do some smoothing"""
        if self.buffer_size is None:
            raise RuntimeError("no buffer")
        with self._lock:
            imgs = [img[None] for img in self._buffer]
        # median = np.median(np.concatenate(imgs, axis=0), axis=0)
        stacked = np.concatenate(imgs, axis=0)
        avg = np.mean(stacked, axis=0)
        std = np.std(stacked, axis=0)
        dims = avg.shape
        avg = avg.reshape(-1)
        avg[std.reshape(-1) > std_threshold] = 0
        img = avg.reshape(*dims)

        if device is not None:
            # If a device is specified, assume we want to move to pytorch
            import torch

            img = torch.FloatTensor(img).to(device)

        return img

    def get_frame(self):
        return self.frame_id

    def get_K(self):
        return self.K.copy()

    def get_info(self):
        return {
            "D": self.D,
            "K": self.K,
            "fx": self.fx,
            "fy": self.fy,
            "px": self.px,
            "py": self.py,
            "near_val": self.near_val,
            "far_val": self.far_val,
            "R": self.R,
            "P": self.P,
            "height": self.height,
            "width": self.width,
        }
