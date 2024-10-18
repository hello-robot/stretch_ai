# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import click
import numpy as np

from stretch.agent import RobotClient

# Mapping and perception
from stretch.core.parameters import get_parameters
from stretch.dynav import RobotAgentMDP


def compute_tilt(camera_xyz, target_xyz):
    """
    a util function for computing robot head tilts so the robot can look at the target object after navigation
    - camera_xyz: estimated (x, y, z) coordinates of camera
    - target_xyz: estimated (x, y, z) coordinates of the target object
    """
    vector = camera_xyz - target_xyz
    return -np.arctan2(vector[2], np.linalg.norm(vector[:2]))


@click.command()
@click.option("--server-ip", "--server_ip", default="127.0.0.1", type=str)
@click.option("--manual-wait", default=False, is_flag=True)
@click.option("--random-goals", default=False, is_flag=True)
@click.option("--explore-iter", default=-1)
@click.option("--re", default=1, type=int)
@click.option(
    "--input-path",
    type=click.Path(),
    default=None,
    help="Input path with default value 'output.npy'",
)
def main(
    server_ip,
    manual_wait,
    navigate_home: bool = False,
    explore_iter: int = 5,
    re: int = 1,
    input_path: str = None,
    **kwargs,
):
    """
    Including only some selected arguments here.

    Args:
        random_goals(bool): randomly sample frontier goals instead of looking for closest
    """
    click.echo("Will connect to a Stretch robot and collect a short trajectory.")
    robot = RobotClient(robot_ip="100.79.44.11")

    print("- Load parameters")
    parameters = get_parameters("dynav_config.yaml")
    # print(parameters)
    if explore_iter >= 0:
        parameters["exploration_steps"] = explore_iter

    print("- Start robot agent with data collection")
    demo = RobotAgentMDP(robot, parameters, server_ip=server_ip, re=re)

    while input("STOP? Y/N") != "Y":
        if input("You want to run manipulation: y/n") == "y":
            text = input("Enter object name: ")
            theta = -0.6
            demo.manipulate(text, theta)

        if input("You want to run placing: y/n") == "y":
            text = input("Enter receptacle name: ")
            theta = -0.6
            demo.place(text, theta)


if __name__ == "__main__":
    main()
