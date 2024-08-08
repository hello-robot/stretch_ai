# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import os


def _ensure_path_exists() -> None:
    """Ensure that the ~/.stretch directory exists."""
    path = os.path.expanduser("~/.stretch")
    if not os.path.exists(path):
        os.makedirs(path)


def lookup_address(robot_ip: str, use_remote_computer: bool = False, update: bool = True) -> str:
    """Return the address of the robot. Will also create and update ~/.stretch/robot_ip.txt file to manage robot IP address.

    Args
        robot_ip: IP address of the robot
        use_remote_computer: Use remote computer or not
        update: Update the robot IP address file
    """
    # Use remote computer or whatever
    _ensure_path_exists()
    if use_remote_computer:
        if len(robot_ip) > 0:
            # Update the robot IP address file
            if update:
                with open(os.path.expanduser("~/.stretch/robot_ip.txt"), "w") as f:
                    f.write(robot_ip)
        else:
            # Look up the robot computer in config directory
            robot_ip = open(os.path.expanduser("~/.stretch/robot_ip.txt")).read().strip()
        recv_address = "tcp://" + robot_ip
    else:
        recv_address = "tcp://127.0.0.1"
    return recv_address
