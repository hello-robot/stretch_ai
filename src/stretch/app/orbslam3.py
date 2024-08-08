# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import argparse
import time

import numpy as np
from pynput import keyboard

import stretch
from stretch.navigation.utils import OrbSlamVisualizer

keys = {"up": False, "down": False, "left": False, "right": False, "escape": False}


def on_press(key):
    if key == keyboard.Key.up:
        keys["up"] = True
    elif key == keyboard.Key.down:
        keys["down"] = True
    elif key == keyboard.Key.left:
        keys["left"] = True
    elif key == keyboard.Key.right:
        keys["right"] = True


def on_release(key):
    if key == keyboard.Key.up:
        keys["up"] = False
    elif key == keyboard.Key.down:
        keys["down"] = False
    elif key == keyboard.Key.left:
        keys["left"] = False
    elif key == keyboard.Key.right:
        keys["right"] = False
    elif key == keyboard.Key.esc:
        keys["escape"] = True


listener = keyboard.Listener(on_press=on_press, on_release=on_release, suppress=True)


def base_keyboard_teleop(apply_translation_vel=0.12, apply_rotational_vel=0.7):
    print("ORBSLAM3 DEMO IS RUNNING")
    print("Use four arrow keys to teleoperate the mobile base around")
    print("Press the 'escape' key to exit")
    while not keys["escape"]:
        translation_vel = apply_translation_vel * int(
            keys["up"]
        ) + -1.0 * apply_translation_vel * int(keys["down"])
        rotational_vel = apply_rotational_vel * int(
            keys["left"]
        ) + -1.0 * apply_rotational_vel * int(keys["right"])
        if translation_vel == 0.0 and rotational_vel == 0.0:
            continue
        stretch.set_base_velocity(translation_vel, rotational_vel)


def feature_matching_dance():
    print("\nINITIALIZING FEATURE MATCHING BY JOGGING HEAD")
    t = np.linspace(0, 2 * np.pi, 40, endpoint=True)
    deltas = 0.05 * np.sin(t)
    for d in deltas:
        stretch.move_by(joint_head_pan=d, joint_head_tilt=d)
        time.sleep(0.05)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ORB-SLAM3 Demo showcasing relative pose estimation computed on imagery from the head nav camera"
    )
    parser.add_argument("vocab_path", type=str, help="ORB-SLAM3 vocabulary path")
    parser.add_argument("config_path", type=str, help="ORB-SLAM3 config path")
    args = parser.parse_args()

    # Initialize ORB-SLAM3 viz
    stretch.connect()
    visualizer = OrbSlamVisualizer(
        args.vocab_path, args.config_path, stretch._client.ip_addr, stretch._client.port
    )
    feature_matching_dance()

    # Initialize base teleop
    listener.start()
    base_keyboard_teleop()
    listener.stop()
