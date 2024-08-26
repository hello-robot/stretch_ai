# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import datetime
import sys
import time
import timeit
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click
import matplotlib.pyplot as plt
import numpy as np
import open3d
import torch
from PIL import Image

# Mapping and perception
from stretch.core.parameters import Parameters, get_parameters
from stretch.dynav import RobotAgentMDP

# Chat and UI tools
from stretch.utils.point_cloud import numpy_to_pcd, show_point_cloud
from stretch.agent import RobotClient

import cv2
import threading

import os

def compute_tilt(camera_xyz, target_xyz):
    '''
        a util function for computing robot head tilts so the robot can look at the target object after navigation
        - camera_xyz: estimated (x, y, z) coordinates of camera
        - target_xyz: estimated (x, y, z) coordinates of the target object
    '''
    vector = camera_xyz - target_xyz
    return -np.arctan2(vector[2], np.linalg.norm(vector[:2]))

@click.command()
@click.option("--ip", default='100.108.67.79', type=str)
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
    ip,
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
    robot = RobotClient(robot_ip = "127.0.0.1")

    print("- Load parameters")
    parameters = get_parameters("dynav_config.yaml")
    # print(parameters)
    if explore_iter >= 0:
        parameters["exploration_steps"] = explore_iter
    object_to_find, location_to_place = None, None

    print("- Start robot agent with data collection")
    demo = RobotAgentMDP(
        robot, parameters, ip = ip, re = re
    )

    while input('STOP? Y/N') != 'Y':
        text = input('Enter object name: ')
        theta = -0.6
        demo.manipulate(text, theta)
            
        if input('You want to run placing: y/n') == 'n':
            continue
        text = input('Enter receptacle name: ')
        theta = -0.6
        demo.place(text, theta)


if __name__ == "__main__":
    main()