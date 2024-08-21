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
import timeit
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import cv2
import numpy as np
import zmq
from overrides import override

import stretch.utils.logger as logger
from stretch.audio.text_to_speech import get_text_to_speech
from stretch.core.comms import CommsNode


class BaseZmqServer(CommsNode, ABC):

    # How often should we print out info about our performance
    report_steps = 100
    fast_report_steps = 10000

    def __init__(
        self,
        send_port: int = 4401,
        recv_port: int = 4402,
        send_state_port: int = 4403,
        send_servo_port: int = 4404,
        use_remote_computer: bool = True,
        verbose: bool = False,
        image_scaling: float = 0.5,
        ee_image_scaling: float = 0.5,  # 0.6,
        depth_scaling: float = 0.001,
        ee_depth_scaling: float = 0.001,
        text_to_speech_engine: str = "gTTS",
    ):
        self.verbose = verbose
        self.context = zmq.Context()
        self.image_scaling = image_scaling
        self.ee_image_scaling = ee_image_scaling
        self.depth_scaling = depth_scaling
        self.ee_depth_scaling = ee_depth_scaling

        # Set up the publisher socket using ZMQ
        self.send_socket = self._make_pub_socket(send_port, use_remote_computer)

        # Publisher for state-only messages (FAST spin rate)
        self.send_state_socket = self._make_pub_socket(send_state_port, use_remote_computer)

        # Publisher for visual servoing images (lower size, faster publishing rate)
        self.send_servo_socket = self._make_pub_socket(send_servo_port, use_remote_computer)

        # Subscriber for actions
        self.recv_socket, self.recv_address = self._make_sub_socket(recv_port, use_remote_computer)
        self._last_step = -1

        # Extensions to the ROS server
        # Text to speech engine - let's let the robot talk
        self.text_to_speech = get_text_to_speech(text_to_speech_engine)

        print("Done setting up connections! Server ready to start.")

        # for the threads
        self.control_mode = "none"
        self._done = False

    def get_control_mode(self) -> str:
        """Get the current control mode of the robot. Can be navigation, manipulation, or none.

        Returns:
            str: The current control mode of the robot.
        """
        if self.client.in_manipulation_mode():
            control_mode = "manipulation"
        elif self.client.in_navigation_mode():
            control_mode = "navigation"
        else:
            control_mode = "none"
        return control_mode

    def _rescale_color_and_depth(
        self, color_image, depth_image, scaling: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Rescale the color and depth images by a given scaling factor."""
        color_image = cv2.resize(
            color_image,
            (0, 0),
            fx=scaling,
            fy=scaling,
            interpolation=cv2.INTER_AREA,
        )
        depth_image = cv2.resize(
            depth_image,
            (0, 0),
            fx=scaling,
            fy=scaling,
            interpolation=cv2.INTER_NEAREST,
        )
        return color_image, depth_image

    def start(self):
        """Starts both threads spinning separately for efficiency."""
        print("==========================================")
        print("Starting up threads:")
        print(" - Starting send thread")
        self._send_thread = threading.Thread(target=self.spin_send)
        print(" - Starting recv thread")
        self._recv_thread = threading.Thread(target=self.spin_recv)
        print(" - Sending state information")
        self._send_state_thread = threading.Thread(target=self.spin_send_state)
        print(" - Sending servo information")
        self._send_servo_thread = threading.Thread(target=self.spin_send_servo)
        self._done = False
        print("Running all...")
        self._send_thread.start()
        self._recv_thread.start()
        self._send_state_thread.start()
        self._send_servo_thread.start()

    @abstractmethod
    def spin_send(self):
        """Spin the send thread."""
        pass

    @abstractmethod
    def spin_recv(self):
        """Spin the receive thread."""
        pass

    @abstractmethod
    def spin_send_state(self):
        """Spin the send state thread."""
        pass

    @abstractmethod
    def spin_send_servo(self):
        """Spin the send servo thread."""
        pass

    def __del__(self):
        self._done = True
        # Wait for the threads to finish
        time.sleep(0.15)

        # Close threads
        self._send_thread.join()
        self._recv_thread.join()
        self._send_state_thread.join()
        self._send_servo_thread.join()

        # Close sockets
        self.recv_socket.close()
        self.send_socket.close()
        self.send_state_socket.close()
        self.context.term()
