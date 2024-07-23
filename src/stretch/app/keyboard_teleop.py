#!/usr/bin/env python

import os
import sys

import click

from stretch.agent.zmq_client import HomeRobotZmqClient

# For Windows
if os.name == "nt":
    import msvcrt

# For Unix (Linux, macOS)
else:
    import termios
    import tty


def key_pressed(robot: HomeRobotZmqClient, key):
    xyt = robot.get_base_pose()
    print(f"Key '{key}' was pressed {xyt}")
    if key == "a":
        xyt[2] += 0.1
    elif key == "d":
        xyt[2] -= 0.1


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
def main(robot_ip: str = "192.168.1.15", local: bool = False):

    # Create robot
    robot = HomeRobotZmqClient(
        robot_ip=robot_ip,
        use_remote_computer=(not local),
    )
    if not robot.in_navigation_mode():
        robot.move_to_nav_posture()

    print("Press W, A, S, or D. Press 'q' to quit.")
    while True:
        char = getch()
        if char == "q":
            break
        elif char in ["w", "a", "s", "d"]:
            key_pressed(robot, char)
        else:
            print("Invalid input. Please press W, A, S, or D. Press 'q' to quit.")


if __name__ == "__main__":
    main()
