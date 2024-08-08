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

import sys
import time

from stretch.navigation.orbslam import OrbSlam


class OrbSlamVisualizer:
    """Visualizer for ORB-SLAM3"""

    def __init__(
        self,
        vocab_path: str = "",
        config_path: str = "",
        camera_ip_addr: str = "",
        camera_port: int = -1,
    ):
        """
        Constructor for Visualizer class.

        Parameters:
        vocab_path (str): ORB-SLAM3 vocabulary path.
        config_path (str): ORB-SLAM3 config path.
        camera_ip_addr (str): Camera's (Head) ZMQ IP address.
        camera_port (int): Camera's (Head) ZMQ port.
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
        assert camera_port != -1, "Camera's ZMQ port must be set."

        self.vocab_path = vocab_path
        self.config_path = config_path
        self.camera_ip_addr = camera_ip_addr
        self.camera_port = int(camera_port)

        self.orbslam = OrbSlam(
            self.vocab_path, self.config_path, self.camera_ip_addr, self.camera_port
        )
        self.orbslam.set_use_viewer(True)
        self.orbslam.initialize()
        self.orbslam.start()

    def start(self):
        # Fetch trajectory points every 0.2 seconds
        while True:
            print(self.orbslam.get_pose())
            time.sleep(0.2)
