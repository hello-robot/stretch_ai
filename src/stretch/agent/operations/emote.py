# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

from time import sleep

import numpy as np

import stretch.motion.constants as constants
from stretch.agent.base import ManagedOperation


class WaveOperation(ManagedOperation):
    """
    Waves the robot's hand
    """

    def can_start(self) -> bool:
        return True

    def run(
        self,
        n_waves: int = 3,
        pitch: float = 0.2,
        yaw_amplitude: float = 0.25,
        roll_amplitude: float = 0.15,
        lift_height: float = 1.0,
    ):
        """
        Waves the robot's wrist.

        Parameters:
            n_waves (int): The number of waves to perform.
            pitch (float): The pitch of the wrist (radians).
            yaw_amplitude (float): The amplitude of the yaw (radians).
            wave_duration (float): The duration of each wave (radians).
        """
        self.robot.switch_to_manipulation_mode()

        assert self.robot.in_manipulation_mode(), "Did not switch to manipulation mode"

        first_pose = [0.0, lift_height, 0.05, 0.0, 0.0, 0.0]

        # move to initial lift height
        first_pose[1] = lift_height
        self.robot.arm_to(first_pose, head=constants.look_at_ee, blocking=True)

        # generate poses
        wave_poses = np.zeros((n_waves * 2, 6))
        for i in range(n_waves):
            j = i * 2
            wave_poses[j] = [0.0, lift_height, 0.05, -yaw_amplitude, pitch, -roll_amplitude]
            wave_poses[j + 1] = [0.0, lift_height, 0.05, yaw_amplitude, pitch, roll_amplitude]

        assert self.robot.in_manipulation_mode(), "Did not switch to manipulation mode"

        # move to poses w/o blocking to make smoother motions
        for pose in wave_poses:
            self.robot.arm_to(pose, head=constants.look_at_ee, blocking=False)
            sleep(0.375)

        self.robot.arm_to(first_pose, head=constants.look_at_ee, blocking=True)

    def was_successful(self) -> bool:
        return True


class NodHeadOperation(ManagedOperation):
    """
    Nods the robot's head.
    """

    def can_start(self) -> bool:
        return True

    def run(
        self,
        n_nods: int = 2,
        pitch_amplitude: float = 0.3,
        override_current_tilt: bool = False,
        tilt_zero=0.0,
    ):
        self.robot.switch_to_manipulation_mode()

        current_pan = self.robot.get_observation().joint[9]
        if not override_current_tilt:
            tilt_zero = self.robot.get_observation().joint[10]

        first_head_pose = [current_pan, tilt_zero]
        self.robot.head_to(*first_head_pose, blocking=True)

        #  generate poses
        poses = np.zeros((n_nods * 2, 2))
        for i in range(n_nods):
            j = i * 2
            poses[j] = [current_pan, tilt_zero - pitch_amplitude]
            poses[j + 1] = [current_pan, tilt_zero]

        for pose in poses:
            self.robot.head_to(*pose, blocking=True)

    def was_successful(self) -> bool:
        return True


class ShakeHeadOperation(ManagedOperation):
    """
    Shakes the robot's head.
    """

    def can_start(self) -> bool:
        return True

    def run(
        self,
        n_shakes: int = 2,
        pan_amplitude: float = 0.3,
        override_current_pan: bool = False,
        pan_zero=-np.pi / 2.0,
    ):
        self.robot.switch_to_manipulation_mode()

        current_tilt = self.robot.get_observation().joint[10]
        if not override_current_pan:
            pan_zero = self.robot.get_observation().joint[9]

        first_head_pose = [pan_zero, current_tilt]
        self.robot.head_to(*first_head_pose, blocking=True)

        # generate poses
        poses = np.zeros((n_shakes * 2, 2))
        for i in range(n_shakes):
            j = i * 2
            poses[j] = [pan_zero - pan_amplitude, current_tilt]
            poses[j + 1] = [pan_zero + pan_amplitude, current_tilt]

        for pose in poses:
            self.robot.head_to(*pose, blocking=True)

        self.robot.head_to(*first_head_pose, blocking=False)

    def was_successful(self) -> bool:
        return True


class ShrugOperation(ManagedOperation):
    """
    "Shrug" the robot's lift.
    """

    def can_start(self) -> bool:
        return True

    def run(self, lift_delta: float = 0.1):
        self.robot.switch_to_manipulation_mode()

        # get current lift state
        joint_state = self.robot.get_observation().joint
        lift = joint_state[3]

        # generate poses
        first_pose = [0.0, lift, 0.05, 0.0, 0.0, 0.0]
        shrug_pose = [0.0, lift + lift_delta, 0.05, 0.0, 0.0, 0.0]

        for pose in [first_pose, shrug_pose, first_pose]:
            self.robot.arm_to(pose, head=constants.look_at_ee, blocking=True)

    def was_successful(self) -> bool:
        return True


class AvertGazeOperation(ManagedOperation):
    """
    Bashfully avert gaze.
    """

    def can_start(self) -> bool:
        return True

    def run(
        self,
        target_tilt: float = -np.pi / 2.5,
        return_to_initial_tilt: bool = False,
        return_to_initial_tilt_delay: float = 1.0,
    ):
        self.robot.switch_to_manipulation_mode()
        joint_state = self.robot.get_observation().joint
        current_pan = joint_state[9]
        current_tilt = joint_state[10]

        avert_pose = [current_pan, target_tilt]
        self.robot.head_to(*avert_pose, blocking=True)

        if return_to_initial_tilt:
            sleep(return_to_initial_tilt_delay)
            self.robot.head_to(current_pan, current_tilt, blocking=False)

    def was_successful(self) -> bool:
        return True


class ApproachOperation(ManagedOperation):
    """
    Move closer to something.
    """

    def can_start(self) -> bool:
        return False

    def run(self):
        raise NotImplementedError

    def was_successful(self) -> bool:
        return False


class WithdrawOperation(ManagedOperation):
    """
    Move away from something.
    """

    def can_start(self) -> bool:
        return False

    def run(self):
        raise NotImplementedError

    def was_successful(self) -> bool:
        return False


class TestOperation(ManagedOperation):
    """
    Wiggles wrist. Demonstrated weird blocking behavior that has since been patched.
    """

    # Print out extra debug information
    debug_arm_to: bool = True

    def can_start(self) -> bool:
        """We can always start this operation."""
        return True

    def run(self):
        self.robot.switch_to_manipulation_mode()

        first_pose = [0.0, 0.75, 0.05, 0.0, 0.0, 0.0]
        self.robot.arm_to(first_pose, head=constants.look_at_ee, lookblocking=True)

        for i in range(5):
            sign = 1 if i % 2 == 0 else -1
            pose = [0.0, 0.75, 0.05, 0.2 * sign, 0.0, 0.0]
            print(f"Moving to pose {i + 1} = {pose}")
            self.robot.arm_to(pose, blocking=True, verbose=self.debug_arm_to)
            sleep(0.5)

    def was_successful(self) -> bool:
        """Successful if the full sequence was completed."""
        # Todo; check if the arm is in the correct position
        return True
