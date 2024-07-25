#!/usr/bin/env python

import os
import sys

import click

from stretch.agent.zmq_client import HomeRobotZmqClient
from stretch.utils import LoopStats


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

    loop = LoopStats("servo_timing", target_loop_rate=15.0)
    counter = 0
    while True:
        loop.mark_start()
        observation = robot.get_servo_observation()
        loop.mark_end()
        loop.pretty_print()
        counter += 1
        if counter % 100 == 0:
            loop.generate_rate_histogram()
        else:
            loop.sleep()


if __name__ == "__main__":
    main()
