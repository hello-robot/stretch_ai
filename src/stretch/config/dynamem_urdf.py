# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

"""
    Dynamem uses OK-Robot's manipulation stack, so it relies on an accurately calibrated urdf.
    This script does the same thing as described in https://github.com/ok-robot/ok-robot/blob/main/docs/robot-calibration.md
"""

import argparse
import xml.etree.ElementTree as ET

parser = argparse.ArgumentParser()
parser.add_argument(
    "--urdf-path",
    default="src/stretch/config/urdf/stretch.urdf",
    help="URDF path",
)
cfgs = parser.parse_args()

# Read robot model from urdf path
tree = ET.parse(cfgs.urdf_path)
root = tree.getroot()

# Check whether joint fake has already been added
fake_exist = False
for joint in root.findall("joint"):
    if joint.get("name") == "joint_fake":
        fake_exist = True

# Add joint fake if not already added
if not fake_exist:
    xml_snippet = """
    
        <link name="fake_link_x">
            <inertial>
                <origin rpy="0.0 0.0 0." xyz="0. 0. 0."/>
                <mass value="0.749143203376"/>
                <inertia ixx="0.0709854511955" ixy="-0.00433428742758" ixz="-0.000186110788698" iyy="0.000437922053343" iyz="-0.00288788257713" izz="0.0711048085017"/>
            </inertial>
        </link>
        <joint name="joint_fake" type="prismatic">
            <origin rpy="0. 0. 0." xyz="0. 0. 0."/>
            <axis xyz="1.0 0.0 0.0"/>
            <parent link="base_link"/>
            <child link="fake_link_x"/>
            <limit effort="100.0" lower="-1.0" upper="1.1" velocity="1.0"/>
        </joint>
    """

    snippet_root = ET.fromstring(f"<root>{xml_snippet}</root>")

    for element in snippet_root:
        root.append(element)

# Change the parent link of joint mast to fake_link_x
for joint in root.findall("joint"):
    if joint.get("name") == "joint_mast":
        parent = joint.find("parent")
        if parent is not None:
            parent.set("link", "fake_link_x")
            break

# Write the resulting urdf file
tree.write(cfgs.urdf_path, xml_declaration=True, encoding="utf-8")
