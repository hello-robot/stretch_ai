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

from stretch.core.comms import CommsNode
from stretch.utils.logger import Logger

logger = Logger(__name__)

try:
    from stretch.audio.text_to_speech import get_text_to_speech

    imported_text_to_speech = True
except ImportError:
    logger.error("Could not import text to speech")
    imported_text_to_speech = False


class BaseZmqServer(CommsNode, ABC):

    # How often should we print out info about our performance
    report_steps = 1000
    fast_report_steps = 10000
    servo_report_steps = 1000
    skip_duplicate_steps: bool = True

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
        if imported_text_to_speech:
            self.text_to_speech = get_text_to_speech(text_to_speech_engine)
        else:
            self.text_to_speech = None

        print("Done setting up connections! Server ready to start.")

        # for the threads
        self.control_mode = "none"
        self._done = False

    @property
    def done(self) -> bool:
        """Check if the server is done."""
        return self._done

    @property
    def running(self) -> bool:
        """Check if the server is running."""
        return not self.done

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

    @property
    def in_manipulation_mode(self) -> bool:
        """Check if the robot is in manipulation mode."""
        return self.control_mode == "manipulation"

    @property
    def in_navigation_mode(self) -> bool:
        """Check if the robot is in navigation mode."""
        return self.control_mode == "navigation"

    # ==================================================================
    # BEGIN: Implement the following methods

    @abstractmethod
    def get_control_mode(self) -> str:
        """Get the control mode of the robot."""
        pass

    @abstractmethod
    def handle_action(self, action: Dict[str, Any]):
        """Handle the action received from the client."""
        pass

    @abstractmethod
    def get_full_observation_message(self) -> Dict[str, Any]:
        """Get the full observation message for the robot. This includes the full state of the robot, including images and depth images."""
        pass

    @abstractmethod
    def get_state_message(self) -> Dict[str, Any]:
        """Get the state message for the robot. This is a smalll message that includes floating point information and booleans like if the robot is homed."""
        pass

    @abstractmethod
    def get_servo_message(self) -> Dict[str, Any]:
        """Get messages for e2e policy learning and visual servoing. These are images and depth images, but lower resolution than the large full state observations, and they include the end effector camera."""
        pass

    @abstractmethod
    def is_running(self) -> bool:
        """Check if the server is running. Will be used to make sure inner loops terminate.

        Returns:
            bool: True if the server is running, False otherwise."""
        pass

    # DONE: Implement the following methods
    # ==================================================================

    def spin_send(self):
        """Send the full state of the robot to the client."""

        # Create a stretch client to get information
        sum_time: float = 0
        steps: int = 0
        t0 = timeit.default_timer()
        print("Starting to send full state")
        while self.is_running():
            data = self.get_full_observation_message()

            # Skip if no data - could not access camera yet
            if data is None:
                continue

            if steps == 0:
                logger.info(f"[SEND LARGE IMAGE STATE] message keys: {data.keys()}")

            self.send_socket.send_pyobj(data)

            # Finish with some speed info
            t1 = timeit.default_timer()
            dt = t1 - t0
            sum_time += dt
            steps += 1
            t0 = t1
            if self.verbose or steps % self.report_steps == 0:
                print(f"[SEND FULL STATE] time taken = {dt} avg = {sum_time/steps}")

            time.sleep(1e-4)
            t0 = timeit.default_timer()

    def spin_recv(self):
        """Receive actions from the client and handle them."""
        sum_time: float = 0
        steps = 0
        t0 = timeit.default_timer()
        while self.is_running():
            try:
                action = self.recv_socket.recv_pyobj(flags=zmq.NOBLOCK)
            except zmq.Again:
                if self.verbose:
                    logger.warning(" - no action received")
                action = None
            if self.verbose:
                logger.info(f" - {self.control_mode=}")
                logger.info(f" - prev action step: {self._last_step}")
            if action is not None:
                if self.verbose:
                    logger.info(f" - Action received: {action}")
                # Tracking step number -- should never go backwards
                action_step = action.get("step", -1)
                if self.skip_duplicate_steps and action_step <= self._last_step:
                    logger.warning(
                        f"Skipping duplicate action {action_step}, last step = {self._last_step}"
                    )
                    continue
                self.handle_action(action)
                self._last_step = max(action_step, self._last_step)
                logger.info(
                    f"Action #{self._last_step} received:",
                    [str(key) for key in action.keys()],
                )
                if self.verbose:
                    logger.info(f" - last action step: {self._last_step}")
            # Finish with some speed info
            t1 = timeit.default_timer()
            dt = t1 - t0
            sum_time += dt
            steps += 1
            t0 = t1
            if self.verbose or steps % self.fast_report_steps == 0:
                logger.info(f"[RECV] time taken = {dt} avg = {sum_time/steps}")

            time.sleep(1e-4)
            t0 = timeit.default_timer()

    def spin_send_state(self):
        """Send a faster version of the state for tracking joint states and robot base"""
        # Create a stretch client to get information
        sum_time: float = 0
        steps: int = 0
        t0 = timeit.default_timer()
        while self.is_running():
            message = self.get_state_message()

            # Skip if no message - could not access or other core information yet
            if message is None:
                continue

            if steps == 0:
                logger.info(f"[SEND MINIMAL STATE] message keys: {message.keys()}")

            self.send_state_socket.send_pyobj(message)

            # Finish with some speed info
            t1 = timeit.default_timer()
            dt = t1 - t0
            sum_time += dt
            steps += 1
            t0 = t1
            if self.verbose or steps % self.fast_report_steps == 0:
                logger.info(f"[SEND FAST STATE] time taken = {dt} avg = {sum_time/steps}")

            time.sleep(1e-4)
            t0 = timeit.default_timer()

    def spin_send_servo(self):
        """Send the images here as well; smaller versions designed for better performance."""
        sum_time: float = 0
        steps: int = 0
        t0 = timeit.default_timer()

        while not self._done:
            message = self.get_servo_message()

            # Skip if no message - could not access camera yet
            if message is None:
                continue

            if steps == 0:
                logger.info(f"[SEND SERVO STATE] message keys: {message.keys()}")

            self.send_servo_socket.send_pyobj(message)

            # Finish with some speed info
            t1 = timeit.default_timer()
            dt = t1 - t0
            sum_time += dt
            steps += 1
            t0 = t1
            if self.verbose or steps % self.servo_report_steps == 1:
                logger.info(
                    f"[SEND SERVO STATE] time taken = {dt} avg = {sum_time/steps} rate={1/(sum_time/steps)}"
                )

            time.sleep(1e-5)
            t0 = timeit.default_timer()

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
