#!/usr/bin/env python

# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import os
import sys

import click
import cv2
import numpy as np

from stretch.agent.zmq_client import HomeRobotZmqClient

# For Windows
if os.name == "nt":
    import msvcrt

# For Unix (Linux, macOS)
else:
    import termios
    import tty


def key_pressed(robot: HomeRobotZmqClient, key):
    """Handle a key press event. Will just move the base for now.

    Args:
        robot (HomeRobotZmqClient): The robot client object.
        key (str): The key that was pressed.
    """
    xyt = robot.get_base_pose()
    print(f"Key '{key}' was pressed {xyt}")
    goal_xyt = np.array([0.0, 0.0, 0.0])
    if key == "w":
        goal_xyt[0] = 0.1
    elif key == "s":
        goal_xyt[0] = -0.1
    elif key == "a":
        goal_xyt[2] = 0.2
    elif key == "d":
        goal_xyt[2] = -0.2
    robot.move_base_to(goal_xyt, relative=True)


def getch():
    if os.name == "nt":  # Windows
        return msvcrt.getch().decode("utf-8").lower()
    else:  # Unix-like
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch.lower()


@click.command()
@click.option("--robot_ip", default="", help="IP address of the robot")
@click.option(
    "--local",
    is_flag=True,
    help="Set if we are executing on the robot and not on a remote computer",
)
@click.option("--headless", is_flag=True, help="Do not show camera feeds")
def main(robot_ip: str = "192.168.1.15", local: bool = False, headless: bool = False):

    # Create robot
    robot = HomeRobotZmqClient(
        robot_ip=robot_ip,
        use_remote_computer=(not local),
    )
    if not robot.in_navigation_mode():
        robot.move_to_nav_posture()

    print("Press W, A, S, or D. Press 'q' to quit.")
    while True:

        if not headless:
            # Get servo observation from the robot
            # This is a more compact, lower-res image representation for better performance over network
            observation = robot.get_servo_observation()

            # Get RGB and Depth
            rgb = observation.rgb
            depth = observation.depth

            # Convert depth to a color image using OpenCV
            depth = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET)

            # RGB to BGR for OpenCV
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            # Concatenate rgb and depth so they are side-by-side
            combined = np.concatenate((rgb, depth), axis=1)

            # Show the image
            cv2.imshow("Robot View", combined)
            char = chr(cv2.waitKey(1) & 0xFF)  # 0xFF is a mask to get the last 8 bits
        else:
            char = getch()

        if char == "q":
            robot.stop()
            break
        elif char in ["w", "a", "s", "d"]:
            key_pressed(robot, char)
        elif not headless and char == chr(255):
            # Nothing was pressed
            pass
        else:
            print(f"Invalid input {char}. Please press W, A, S, or D. Press 'q' to quit.")


if __name__ == "__main__":
    main()
