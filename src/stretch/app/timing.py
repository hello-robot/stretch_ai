#!/usr/bin/env python

import os
import sys

import click
import cv2
import numpy as np

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
            cv2.waitKey(1)

        loop.mark_end()
        loop.pretty_print()
        counter += 1
        if counter % 100 == 0:
            loop.generate_rate_histogram()
        else:
            loop.sleep()


if __name__ == "__main__":
    main()
