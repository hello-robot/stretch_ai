# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import os
import shutil
from datetime import datetime

path = os.path.expanduser("~/.stretch")


def _ensure_path_exists() -> None:
    """Ensure that the ~/.stretch directory exists."""

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


def get_path_to_map(name: str) -> str:
    """Gets a map filename in the .stretch store directory"""
    _ensure_path_exists()
    os.makedirs(os.path.join(path, "maps"), exist_ok=True)
    return os.path.join(path, "maps", name)


def get_path_to_image(name: str) -> str:
    """Gets an image filename in the .stretch store directory"""
    _ensure_path_exists()
    os.makedirs(os.path.join(path, "images"), exist_ok=True)
    return os.path.join(path, "images", name)


def get_path_to_debug(name: str) -> str:
    """Gets a debug filename in the .stretch store directory"""
    _ensure_path_exists()
    os.makedirs(os.path.join(path, "debug"), exist_ok=True)
    return os.path.join(path, "debug", name)


def get_path_to_default_credentials() -> str:
    """Gets the path to the default credentials file"""
    return os.path.join(path, "credentials.json")


def get_path_to_saved_map() -> str:
    """Gets the path to the saved map file"""
    return os.path.join(path, "map.pkl")


def get_path_to_backup_saved_map(timestamp: str) -> str:
    """Backup the saved map file"""
    return os.path.join(path, "backup", f"map_{timestamp}.pkl")


def backup_saved_map() -> bool:
    """Backup the saved map file to a new location.

    Returns:
        True if the file is backed up, False otherwise
    """

    # Create a timestamp for the backup
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    # Ensure the path exists
    _ensure_path_exists()

    # Create a backup directory if it doesn't exist
    os.makedirs(os.path.join(path, "backup"), exist_ok=True)

    # Check to see if a file exists first
    if os.path.exists(get_path_to_saved_map()):
        shutil.copyfile(get_path_to_saved_map(), get_path_to_backup_saved_map(timestamp))
        return True
    return False
