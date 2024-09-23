# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import threading
import time
from typing import List

import numpy as np
import orbslam3

from stretch.drivers.d435 import D435i
from stretch.navigation.base import Pose, Slam
from stretch.navigation.utils.geometry import transformation_matrix_to_pose


class OrbSlam(Slam):
    def __init__(
        self,
        vocab_path: str = "",
        config_path: str = "",
        camera_ip_addr: str = "",
        base_port: int = -1,
    ):
        """
        Constructor for Slam class.
        Creates ORB-SLAM3 backend and camera sockets.

        Parameters:
        vocab_path (str): ORB-SLAM3 vocabulary path.
        config_path (str): ORB-SLAM3 config path.
        camera_ip_addr (str): Camera's (Head) ZMQ IP address.
        base_port (int): Camera's (Head) ZMQ port.
        """
        assert (
            vocab_path != ""
        ), "Vocabulary path should not be \
                                 an empty string."
        assert (
            config_path != ""
        ), "ORB-SLAM3 config file path should not be \
                                   an empty string."
        assert camera_ip_addr != "", "Camera's ZMQ IP address must be set."
        assert base_port != -1, "Camera's ZMQ port must be set."

        self.vocab_path = vocab_path
        self.config_path = config_path

        self.slam_system = orbslam3.System(self.vocab_path, self.config_path, orbslam3.Sensor.RGBD)
        self.slam_system.set_use_viewer(False)
        self.camera = D435i()

        self.color_image = None
        self.depth_image = None
        self.dirty = False
        self.timestamp = None
        self.pose = Pose(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self.trajectory_points: List[Pose] = []

    def __del__(self):
        self.camera_thread.join()
        self.slam_thread.join()

    def initialize(self):
        """
        Initializes ORB-SLAM3 backend.
        """
        self.slam_system.initialize()

    def set_use_viewer(self, use_viewer: bool = True):
        """
        Use ORB_SLAM3's visualization GUI. Must be called before initialize()

        Parameters:
        use_viewer (bool): Flag to set or unset the GUI. Defaults to True.
        """
        self.slam_system.set_use_viewer(use_viewer)

    def get_pose(self) -> Pose:
        """
        Get latest camera pose along with timestamp.

        Returns:
        Pose: Timestamp and 6-DOF pose in the camera frame.
        Formatted as (timestamp, x, y, z, roll, pitch, yaw).
        Distances are in meters and euler angles are in radians.
        """
        return self.pose

    def get_trajectory_points(self) -> List[Pose]:
        """
        Get camera trajectory points as a list.

        Returns:
        List[Pose]: List of all camera trajectory points.
        """
        return self.trajectory_points

    def camera_thread(self):
        """
        Camera listener thread.
        """
        while True:
            msg = self.camera.get_message()
            self.color_image = np.array(msg["color_image"])
            self.depth_image = np.array(msg["depth_image"])
            self.timestamp = float(time.time())
            self.dirty = True

    def slam_thread(self):
        """
        SLAM system thread. TODO: Rate-limiting is currently not implemented.
        """
        while True:
            if self.dirty:
                Tcw = self.slam_system.process_image_rgbd(
                    self.color_image, self.depth_image, [], self.timestamp
                )
                self.dirty = False

                # Tcw is a 4x4 transformation matrix unrolled to a python list
                # in row-major order of 16 elements

                # Reshape it to a 4x4 matrix
                if (len(Tcw)) == 0:
                    continue

                Tcw = np.array(Tcw).reshape(4, 4)

                # Compute Twc
                Twc = np.linalg.inv(Tcw)

                self.pose = transformation_matrix_to_pose(Twc)
                self.trajectory_points.append(self.pose)

    def start(self):
        """
        Start mapping and localization.
        Creates threads for listening incoming image frames and
        processing them through the SLAM pipeline.
        """
        self.camera_thread = threading.Thread(target=self.camera_thread, args=())
        self.slam_thread = threading.Thread(target=self.slam_thread, args=())

        self.camera_thread.start()
        self.slam_thread.start()
