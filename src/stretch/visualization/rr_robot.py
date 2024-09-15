#!/usr/bin/env python3

# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import importlib.resources as importlib_resources

import numpy as np
import urchin as urdf_loader

pkg_path = str(importlib_resources.files("stretch_urdf"))
model_name = "SE3"  # RE1V0, RE2V0, SE3
tool_name = "eoa_wrist_dw3_tool_sg3"  # eoa_wrist_dw3_tool_sg3, tool_stretch_gripper, etc
urdf_file_path = pkg_path + f"/{model_name}/stretch_description_{model_name}_{tool_name}.urdf"
mesh_files_directory_path = pkg_path + f"/{model_name}/meshes"


class URDFmodel:
    def __init__(self) -> None:
        """
        Load URDF model
        """
        self.urdf = urdf_loader.URDF.load(urdf_file_path, lazy_load_meshes=True)
        self.joints_names = [
            "joint_wrist_yaw",
            "joint_wrist_pitch",
            "joint_wrist_roll",
            "joint_lift",
            "joint_arm_l0",
            "joint_arm_l1",
            "joint_arm_l2",
            "joint_arm_l3",
            "joint_head_pan",
            "joint_head_tilt",
            "joint_gripper_finger_left",
        ]

    def get_transform(self, cfg: dict, link_name: str) -> np.ndarray:
        """
        Get transformation matrix of the link w.r.t. the base_link
        """
        lk_cfg = {
            "joint_wrist_yaw": cfg["wrist_yaw"],
            "joint_wrist_pitch": cfg["wrist_pitch"],
            "joint_wrist_roll": cfg["wrist_roll"],
            "joint_lift": cfg["lift"],
            "joint_arm_l0": cfg["arm"] / 4,
            "joint_arm_l1": cfg["arm"] / 4,
            "joint_arm_l2": cfg["arm"] / 4,
            "joint_arm_l3": cfg["arm"] / 4,
            "joint_head_pan": cfg["head_pan"],
            "joint_head_tilt": cfg["head_tilt"],
        }
        if "gripper" in cfg.keys():
            lk_cfg["joint_gripper_finger_left"] = cfg["gripper"]
            lk_cfg["joint_gripper_finger_right"] = cfg["gripper"]
        return self.urdf.link_fk(lk_cfg, link=link_name)
