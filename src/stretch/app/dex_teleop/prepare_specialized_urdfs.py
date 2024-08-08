# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import pathlib
import pprint
from copy import deepcopy

import dex_teleop_parameters as dt
import numpy as np
from urdf_parser_py import urdf as ud


def save_urdf_file(robot, file_name):
    urdf_string = robot.to_xml_string()
    print("Saving new URDF file to", file_name)
    fid = open(file_name, "w")
    fid.write(urdf_string)
    fid.close()
    print("Finished saving")


use_on_robot = True

if use_on_robot:
    import stretch_body.hello_utils as hu

    calibration_dir = pathlib.Path(hu.get_fleet_directory()) / "exported_urdf"
    urdf_path = calibration_dir / "stretch.urdf"
    controller_params_path = calibration_dir / "controller_calibration_head.yaml"
    urdf_filename = str(urdf_path.absolute())
else:
    urdf_filename = "./stretch.urdf"


non_fixed_joints = [
    "joint_lift",
    "joint_arm_l0",
    "joint_arm_l1",
    "joint_arm_l2",
    "joint_arm_l3",
    "joint_wrist_yaw",
    "joint_wrist_pitch",
    "joint_wrist_roll",
]


# Enables more conservative joint limits to be set than in the
# original URDF. Joint limits that are outside the originally
# permitted range will be clipped to the original range. Joint limits
# with a value of None will be set to the original limit.

use_original_limits = True  # False

if use_original_limits:
    # Beware of gimbal lock if joint_wrist_pitch is too close to -90 deg

    ik_joint_limits = {
        "joint_mobile_base_translation": (None, None),
        "joint_mobile_base_rotation": (None, None),
        "joint_lift": (None, None),
        "joint_arm_l0": (None, None),
        "joint_wrist_yaw": (None, None),
        "joint_wrist_pitch": (dt.wrist_pitch_lower_limit, None),
        "joint_wrist_roll": (None, None),
    }

    #'joint_wrist_pitch' : (None, None), # default lowest pitch is -1.57 rad on the default URDF (-89.954373836 deg)~

else:

    ik_joint_limits = {
        "joint_mobile_base_translation": (-0.25, 0.25),
        "joint_mobile_base_rotation": (-(np.pi / 2.0), np.pi / 2.0),
        "joint_lift": (0.01, 1.09),
        "joint_arm_l0": (0.01, 0.48),
        "joint_wrist_yaw": (-(np.pi / 4.0), np.pi),
        "joint_wrist_pitch": (-0.9 * (np.pi / 2.0), np.pi / 20.0),
        "joint_wrist_roll": (-(np.pi / 2.0), np.pi / 2.0),
    }

    print("Setting new joint limits with ik_joint_limits =")
    pprint.pprint(ik_joint_limits)


# Load and store the original uncalibrated URDF.
print()
print("Loading URDF from:")
print(urdf_filename)
print("The specialized URDFs will be derived from this URDF.")
robot = ud.Robot.from_xml_file(urdf_filename)

# Change any joint that should be immobile for end effector IK into a fixed joint
for j in robot.joint_map.keys():
    if j not in non_fixed_joints:
        joint = robot.joint_map[j]
        # print('(joint name, joint type) =', (joint.name, joint.type))
        joint.type = "fixed"

# Replace telescoping arm with a single prismatic joint

# arm joints from proximal to distal
all_arm_joints = ["joint_arm_l4", "joint_arm_l3", "joint_arm_l2", "joint_arm_l1", "joint_arm_l0"]

prismatic_arm_joints = all_arm_joints[1:]

removed_arm_joints = all_arm_joints[1:-1]

xyz_total = np.array([0.0, 0.0, 0.0])
limit_upper_total = 0.0

for j in prismatic_arm_joints:
    joint = robot.joint_map[j]
    # print(j + ' =', joint)
    xyz = joint.origin.xyz
    # print('xyz =', xyz)
    xyz_total = xyz_total + xyz
    limit_upper = joint.limit.upper
    # print('limit_upper =', limit_upper)
    limit_upper_total = limit_upper_total + limit_upper

# print('xyz_total =', xyz_total)
# print('limit_upper_total =', limit_upper_total)

proximal_arm_joint = robot.joint_map[all_arm_joints[0]]
near_proximal_arm_joint = robot.joint_map[all_arm_joints[1]]
near_distal_arm_joint = robot.joint_map[all_arm_joints[-2]]
distal_arm_joint = robot.joint_map[all_arm_joints[-1]]

# Directly connect the proximal and distal parts of the arm
distal_arm_joint.parent = near_proximal_arm_joint.parent

# Make the distal prismatic joint act like the full arm
distal_arm_joint.origin.xyz = xyz_total
distal_arm_joint.limit.upper = limit_upper_total

# Make the telescoping joints in between immobile
for j in removed_arm_joints:
    joint = robot.joint_map[j]
    joint.type = "fixed"


robot_rotary = robot
robot_prismatic = deepcopy(robot)

###############################################
# ADD VIRTUAL ROTARY JOINT FOR MOBILE BASE

# Add a virtual base link
link_virtual_base_rotary = ud.Link(
    name="virtual_base", visual=None, inertial=None, collision=None, origin=None
)

# Add rotary joint for the mobile base
origin_rotary = ud.Pose(xyz=[0, 0, 0], rpy=[0, 0, 0])

limit_rotary = ud.JointLimit(effort=10, velocity=1, lower=-np.pi, upper=np.pi)

joint_mobile_base_rotation = ud.Joint(
    name="joint_mobile_base_rotation",
    parent="virtual_base",
    child="base_link",
    joint_type="revolute",
    axis=[0, 0, 1],
    origin=origin_rotary,
    limit=limit_rotary,
    dynamics=None,
    safety_controller=None,
    calibration=None,
    mimic=None,
)

robot_rotary.add_link(link_virtual_base_rotary)
robot_rotary.add_joint(joint_mobile_base_rotation)
###############################################


###############################################
# ADD VIRTUAL PRISMATIC JOINT FOR MOBILE BASE

# Add a virtual base link
link_virtual_base_prismatic = ud.Link(
    name="virtual_base", visual=None, inertial=None, collision=None, origin=None
)

# Add rotary joint for the mobile base
origin_prismatic = ud.Pose(xyz=[0, 0, 0], rpy=[0, 0, 0])

limit_prismatic = ud.JointLimit(effort=10, velocity=1, lower=-1.0, upper=1.0)

joint_mobile_base_translation = ud.Joint(
    name="joint_mobile_base_translation",
    parent="virtual_base",
    child="base_link",
    joint_type="prismatic",
    axis=[1, 0, 0],
    origin=origin_prismatic,
    limit=limit_prismatic,
    dynamics=None,
    safety_controller=None,
    calibration=None,
    mimic=None,
)

robot_prismatic.add_link(link_virtual_base_prismatic)
robot_prismatic.add_joint(joint_mobile_base_translation)

###############################################

# When specified, this sets more conservative joint limits than the
# original URDF. Joint limits that are outside the originally
# permitted range are clipped to the original range. Joint limits
# with a value of None are set to the original limit.
for robot in [robot_rotary, robot_prismatic]:
    for j in ik_joint_limits:
        joint = robot.joint_map.get(j, None)
        if joint is not None:

            original_upper = joint.limit.upper
            requested_upper = ik_joint_limits[j][1]
            # print()
            # print('joint =', j)
            # print('original_upper =', original_upper)
            # print('requested_upper =', requested_upper)
            if requested_upper is not None:
                new_upper = min(requested_upper, original_upper)
                # print('new_upper =', new_upper)
                robot.joint_map[j].limit.upper = new_upper
                # print()

            original_lower = joint.limit.lower
            requested_lower = ik_joint_limits[j][0]
            if requested_lower is not None:
                new_lower = max(requested_lower, original_lower)
                robot.joint_map[j].limit.lower = new_lower


# print('************************************************')
# print('after adding link and joint: robot =', robot)
# print('************************************************')

print()
save_urdf_file(robot_rotary, "stretch_base_rotation_ik.urdf")
save_urdf_file(robot_prismatic, "stretch_base_translation_ik.urdf")


# Create versions with fixed wrists

for robot in [robot_rotary, robot_prismatic]:
    print("Prepare URDF with a fixed wrist.")
    non_fixed_joints = [
        "joint_mobile_base_translation",
        "joint_mobile_base_rotation",
        "joint_lift",
        "joint_arm_l0",
    ]

    # Change any joint that should be immobile for end effector IK into a fixed joint
    for j in robot.joint_map.keys():
        if j not in non_fixed_joints:
            joint = robot.joint_map[j]
            # print('(joint name, joint type) =', (joint.name, joint.type))
            joint.type = "fixed"

save_urdf_file(robot_rotary, "stretch_base_rotation_ik_with_fixed_wrist.urdf")
save_urdf_file(robot_prismatic, "stretch_base_translation_ik_with_fixed_wrist.urdf")
