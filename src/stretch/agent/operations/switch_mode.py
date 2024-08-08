# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

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
