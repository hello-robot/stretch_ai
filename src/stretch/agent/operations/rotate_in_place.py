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

from stretch.agent.base import ManagedOperation


class RotateInPlaceOperation(ManagedOperation):
    """Rotate the robot in place. Number of steps is determined by parameters file."""

    def can_start(self) -> bool:
        self.attempt(f"Rotating for {self.parameters['in_place_rotation_steps']} steps.")
        return True

    def run(self) -> None:
        self.intro("rotating for {self.parameters['in_place_rotation_steps']} steps.")
        self._successful = False
        self.robot.rotate_in_place(
            steps=self.parameters["in_place_rotation_steps"],
            visualize=False,
        )
        self._successful = True

    def was_successful(self) -> bool:
        return self._successful
