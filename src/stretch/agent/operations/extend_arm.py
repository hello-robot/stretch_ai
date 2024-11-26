# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import stretch.motion.constants as constants
from stretch.agent.base import ManagedOperation
from stretch.motion.kinematics import HelloStretchIdx


class ExtendArm(ManagedOperation):
    """
    Extend the robots arm at a height that is reasonable for handing over an object.
    """

    _pitch = 0.2
    _lift_height = 0.7
    _arm_extension = 0.2

    def can_start(self) -> bool:
        return True

    def configure(self, pitch: float = 0.2, lift_height: float = 0.9, arm_extension: float = 0.2):
        """Configure the operation."""
        self._pitch = pitch
        self._lift_height = lift_height
        self._arm_extension = arm_extension

    def run(self):
        """
        Raises and extends the arm
        """
        self.robot.switch_to_manipulation_mode()

        assert self.robot.in_manipulation_mode(), "Did not switch to manipulation mode"

        joint_state = self.robot.get_joint_positions()
        lifted_joint_state = joint_state.copy()
        lifted_joint_state[HelloStretchIdx.LIFT] = self._lift_height

        # move to initial lift height
        self.robot.arm_to(lifted_joint_state, head=constants.look_at_ee, blocking=True)
        # self.agent.robot_say("I am extending my arm!")
        # sleep(2.0)

        second_pose = [0.0, self._lift_height, self._arm_extension, 0.0, self._pitch, 0.0]
        self.robot.arm_to(second_pose, head=constants.look_at_ee, blocking=True)

    def was_successful(self) -> bool:
        return True
