import copy
import threading
import time
from typing import List, Tuple

import stretch.comms.recv_head_nav_cam as recv_head_nav_cam
from stretch.navigation.base import Pose, Slam 

import cv2
import numpy as np
import orbslam3

class OrbSlam(Slam):
    def __init__(self, vocab_path: str = "", config_path: str = "",
                 camera_ip_addr: str = "", base_port: int = -1):
        """
        Constructor for Slam class.
        Creates ORB-SLAM3 backend and camera sockets.

        Parameters:
        vocab_path (str): ORB-SLAM3 vocabulary path.
        config_path (str): ORB-SLAM3 config path.
        camera_ip_addr (str): Camera's (Head) ZMQ IP address.
        base_port (int): Camera's (Head) ZMQ port.
        """
        assert vocab_path != "","Vocabulary path should not be \
                                 an empty string."
        assert config_path != "", "ORB-SLAM3 config file path should not be \
                                   an empty string."
        assert camera_ip_addr != "", "Camera's ZMQ IP address must be set."
        assert base_port != -1, "Camera's ZMQ port must be set."

        self.vocab_path = vocab_path
        self.config_path = config_path

        self.slam_system = orbslam3.System(self.vocab_path,
                                           self.config_path,
                                           orbslam3.Sensor.MONOCULAR)
        self.slam_system.set_use_viewer(False)
        _, self.camera_sock = recv_head_nav_cam.initialize(camera_ip_addr,
                                                           base_port+4,
                                                           base_port+5)

        self.image = None
        self.timestamp = None
        self.pose = Pose(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self.trajectory_points = []

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
        Each entry has a timestamp at zero-index followed by a
        row-major unrolled transformation matrix w.r.t the reference frame.
        """
        return self.trajectory_points

    def camera_thread(self):
        """
        Camera listener thread.
        """
        while True:
            self.image = cv2.cvtColor(
                recv_head_nav_cam.recv_imagery_as_base64_str(self.camera_sock),
                cv2.COLOR_RGB2BGR)
            self.timestamp = time.time()

    def slam_thread(self):
        """
        SLAM system thread. TODO: Rate-limiting is currently not implemented.
        """
        while True:
            if type(self.image) == np.ndarray and self.timestamp:
                current_image = copy.deepcopy(self.image)
                self.image = None
                self.slam_system.process_image_mono(current_image,
                                                    self.timestamp)

                self.trajectory_points.clear()
                for point in self.slam_system.get_trajectory_points():
                    ## Extract timestamp and quaternion
                    timestamp = point[0]
                    q = np.array([point[4], point[5], point[6], point[7]])
                    
                    # Extract translation vector
                    twc = np.array([point[1], point[2], point[3]])
                    
                    # Compute Euler angles
                    q /= np.linalg.norm(q)  # Normalize quaternion

                    # Extract quaternion components
                    w, x, y, z = q

                    # Roll (x-axis rotation)
                    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))

                    # Pitch (y-axis rotation)
                    sin_pitch = 2 * (w * y - z * x)
                    if np.abs(sin_pitch) >= 1:
                        pitch = np.sign(sin_pitch) * np.pi / 2
                    else:
                        pitch = np.arcsin(sin_pitch)

                    # Yaw (z-axis rotation)
                    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))

                    self.trajectory_points.append(Pose(timestamp,
                                                       twc[0],
                                                       twc[1],
                                                       twc[2],
                                                       roll,
                                                       pitch,
                                                       yaw))

                if len(self.trajectory_points) > 0:
                    self.pose = self.trajectory_points[-1]

    def start(self):
        """
        Start mapping and localization.
        Creates threads for listening incoming image frames and
        processing them through the SLAM pipeline.
        """
        self.camera_thread = \
            threading.Thread(target=self.camera_thread, args=())
        self.slam_thread = \
            threading.Thread(target=self.slam_thread, args=())

        self.camera_thread.start()
        self.slam_thread.start()
