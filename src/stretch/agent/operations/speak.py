# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

from time import sleep

from stretch.agent.base import ManagedOperation


class SpeakOperation(ManagedOperation):
    """
    Speaks a message given by the user.
    """

    _success = True
    _message = ""
    _sleep_time = 3.0

    def can_start(self) -> bool:
        return True

    def configure(self, message: str = "Hello, world!", sleep_time: float = 3.0):
        """Configure the operation given a message to speak."""
        self._message = message
        self._sleep_time = sleep_time

    def run(
        self,
    ):
        """
        Speaks a message

        Parameters:
            message (str): The message to speak.
        """
        self.agent.robot_say(self._message)
        sleep(self._sleep_time)

    def was_successful(self) -> bool:
        """Return true if successful"""
        return True
