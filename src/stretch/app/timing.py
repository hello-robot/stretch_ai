#!/usr/bin/env python

# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.


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
@click.option(
    "--iterations", default=100, help="Number of iterations between rate histogram updates"
)
def main(
    robot_ip: str = "",
    local: bool = False,
    headless: bool = False,
    iterations: int = 500,
):

    # Create robot
    robot = HomeRobotZmqClient(
        robot_ip=robot_ip,
        use_remote_computer=(not local),
    )

    if not headless:
        print("Press 'q' to quit")

    loop = LoopStats("servo_timing", target_loop_rate=15.0)
    counter = 0
    while True:
        loop.mark_start()
        observation = robot.get_servo_observation()

        char = None
        if not headless:
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

        loop.mark_end()
        loop.pretty_print()
        counter += 1
        if counter % iterations == 0:
            loop.generate_rate_histogram()
        else:
            loop.sleep()

        # Get key press and act on it
        if not headless and char == "q":
            break

    if not headless:
        cv2.destroyAllWindows()
    robot.stop()


if __name__ == "__main__":
    main()
