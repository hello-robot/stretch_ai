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


class GoToNavOperation(ManagedOperation):
    """Put the robot into navigation mode"""

    def can_start(self) -> bool:
        self.attempt("will switch to navigation mode.")
        return True

    def run(self) -> None:
        self.intro("Switching to navigation mode.")
        self.robot.move_to_nav_posture()
        self.info("Robot is in navigation mode: {}".format(self.robot.in_navigation_mode()))

    def was_successful(self) -> bool:
        res = self.robot.in_navigation_mode()
        if res:
            self.cheer("Robot is in navigation mode.")
        else:
            self.error("Robot is still not in navigation mode.")
        return res
