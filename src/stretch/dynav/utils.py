# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import numpy as np


def compute_tilt(camera_xyz, target_xyz):
    """
    a util function for computing robot head tilts so the robot can look at the target object after navigation
    - camera_xyz: estimated (x, y, z) coordinates of camera
    - target_xyz: estimated (x, y, z) coordinates of the target object
    """
    if not isinstance(camera_xyz, np.ndarray):
        camera_xyz = np.array(camera_xyz)
    if not isinstance(target_xyz, np.ndarray):
        target_xyz = np.array(target_xyz)
    vector = camera_xyz - target_xyz
    return -np.arctan2(vector[2], np.linalg.norm(vector[:2]))


def get_mode(mode: str) -> str:
    """Helper function to get the mode from the user input. Mode can be navigation, exploration, manipulation, or save.

    Args:
        mode (str): mode to select

    Returns:
        str: mode
    """

    if mode == "navigation":
        return "N"
    elif mode == "explore":
        return "E"
    elif mode == "manipulation":
        return "manipulation"
    elif mode == "save":
        return "S"
    else:
        mode = None
        print("Select mode: E for exploration, N for open-vocabulary navigation, S for save.")
        while mode is None:
            mode = input("select mode? E/N/S: ")
            if mode == "E" or mode == "N" or mode == "S":
                break
            else:
                print("Invalid mode. Please select again.")
        return mode.upper()
