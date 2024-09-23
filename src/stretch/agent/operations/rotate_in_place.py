# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

from stretch.agent.base import ManagedOperation


class RotateInPlaceOperation(ManagedOperation):
    """Rotate the robot in place. Number of steps is determined by parameters file."""

    def can_start(self) -> bool:
        self.attempt(f"Rotating for {self.parameters['agent']['in_place_rotation_steps']} steps.")
        return True

    def run(self) -> None:
        self.intro("rotating for {self.parameters['agent']['in_place_rotation_steps']} steps.")
        self._successful = False
        self.robot.rotate_in_place(
            steps=self.parameters["agent"]["in_place_rotation_steps"],
            visualize=False,
        )
        self._successful = True

    def was_successful(self) -> bool:
        return self._successful
