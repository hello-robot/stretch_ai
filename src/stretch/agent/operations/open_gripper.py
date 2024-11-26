# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.


from stretch.agent.base import ManagedOperation


class OpenGripper(ManagedOperation):
    """
    Open the robot's gripper.
    """

    def can_start(self) -> bool:
        return True

    def run(self):
        """
        Raises and extends the arm
        """
        self.robot.switch_to_manipulation_mode()

        assert self.robot.in_manipulation_mode(), "Did not switch to manipulation mode"

        self.robot.open_gripper()

    def was_successful(self) -> bool:
        return True
