from time import sleep

import numpy as np

from stretch.agent.base import ManagedOperation


class EmoteOperation(ManagedOperation):
    """
    Provides emote functionality for the robot.
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
        first_pose = [0.0, lift_height, 0.05, 0.0, 0.0, 0.0]

        # move to initial lift height
        first_pose[1] = lift_height
        self.robot.arm_to(first_pose, blocking=False)

        # hacky wait...
        while True:
            joint_state = self.robot.get_observation().joint
            lift = joint_state[3]
            if abs(lift - lift_height) < 0.01:
                break
            sleep(0.05)

        wave_waypoints = np.zeros((n_waves * 2, 6))
        for i in range(n_waves):
            j = i * 2
            wave_waypoints[j] = [0.0, lift_height, 0.05, -yaw_amplitude, pitch, -roll_amplitude]
            wave_waypoints[j + 1] = [0.0, lift_height, 0.05, yaw_amplitude, pitch, roll_amplitude]

        for waypoint in wave_waypoints:
            self.robot.arm_to(waypoint, blocking=False)
            sleep(0.375)

        self.robot.arm_to(first_pose, blocking=False)

    def was_successful(self) -> bool:
        return True
