#!/usr/bin/env python
# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.


# Dynamem frequently shows unstable joint state control.
# It turns out there is latency in robot's head and arm joint state
# Here we simulate how Dynamem uses AnyGrasp to pick up objects for debugging.

import time

import click
import numpy as np

from stretch.agent import RobotClient


@click.command()
@click.option("--local", is_flag=True, help="Run code locally on the robot.")
@click.option("--recv_port", default=4401, help="Port to receive observations on")
@click.option("--send_port", default=4402, help="Port to send actions to on the robot")
@click.option("--robot_ip", default="100.76.87.51")
def main(
    local: bool = True,
    recv_port: int = 4401,
    send_port: int = 4402,
    robot_ip: str = "100.76.87.51",
):
    robot = RobotClient(robot_ip=robot_ip)
    robot.switch_to_manipulation_mode()
    for i in range(2):
        print("-" * 20, "test suite #", i, " ", "-" * 20)
        robot.arm_to(np.array([0, 0.7, 0.1, 0, -0.6, 0]), blocking=True)

        relative_arm_movement = np.zeros(6)
        relative_arm_movement[0] = np.random.uniform(-0.03, 0.03)
        relative_arm_movement[1] = np.random.uniform(-0.05, 0.05)
        relative_arm_movement[2] = np.random.uniform(0.05, 0.2)
        print("Relative movement", relative_arm_movement)

        for j in range(3):
            print("-" * 10, "movement #", j, " ", "-" * 10)

            target_head_pan, target_head_tilt = -0.2, -0.8
            robot.head_to(head_pan=-0.1, head_tilt=-0.3, blocking=True)

            cur_state = robot.get_six_joints()
            new_state = (
                cur_state + relative_arm_movement
                if j == 0
                else cur_state + relative_arm_movement * 0.5
            )
            # robot.arm_to(new_state, head=[target_head_pan, target_head_tilt], blocking=True)
            robot.arm_to(new_state, blocking=True)
            robot.head_to(head_pan=target_head_pan, head_tilt=target_head_tilt, blocking=True)
            time.sleep(0.3)
            final_state = robot.get_six_joints()

            print("Expected", new_state)
            print("Actual", final_state)
            error = final_state - new_state
            print("Lift Error", error[1])
            print("Arm Error", error[2])

            time.sleep(0.3)
        time.sleep(1)


if __name__ == "__main__":
    main()
