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
import re

import numpy as np
import urchin as urdf_loader

from stretch.visualization import rr_urdf

pkg_path = str(importlib_resources.files("stretch_urdf"))
model_name = "SE3"  # RE1V0, RE2V0, SE3
tool_name = "eoa_wrist_dw3_tool_sg3"  # eoa_wrist_dw3_tool_sg3, tool_stretch_gripper, etc
urdf_file_path = pkg_path + f"/{model_name}/stretch_description_{model_name}_{tool_name}.urdf"
mesh_files_directory_path = pkg_path + f"/{model_name}/meshes"
urdf_dir = pkg_path + f"/{model_name}/"


def get_absolute_path_stretch_urdf(urdf_file_path, mesh_files_directory_path) -> str:
    """
    Generates Robot XML with absolute path to mesh files
    Args:
        robot_pose_attrib: Robot pose attributes in form {"pos": "x y z", "quat": "x y z w"}
    Returns:
        str: Path to the generated XML file
    """

    with open(urdf_file_path, "r") as f:
        _urdf_file = f.read()

    # find all the line which has the pattrn {file="something.type"}
    # and replace the file path with the absolute path
    pattern = r'filename="(.+?)"'
    for match in re.finditer(pattern, _urdf_file):
        orig = match.group(1)
        fn = match.group(1).split("/")[-1]
        file_path = mesh_files_directory_path + "/" + fn
        _urdf_file = _urdf_file.replace(orig, file_path)

    # Absosolute path converted streth xml
    temp_abs_urdf = "stretch_temp_abs.urdf"
    with open(urdf_dir + temp_abs_urdf, "w") as f:
        f.write(_urdf_file)
    print("Saving temp abs path stretch urdf: {}".format(urdf_dir + f"{temp_abs_urdf}"))
    return urdf_dir + temp_abs_urdf


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


if __name__ == "__main__":
    urdf_file = get_absolute_path_stretch_urdf(urdf_file_path, mesh_files_directory_path)
    urdf_logger = rr_urdf.URDFLogger(urdf_file)
    urdf_logger.log()
