# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

###########################################################################
# The following license applies to simple_ik.py and
# simple_ik_equations_numba.py (the "Files"), which contain software
# for use with the Stretch mobile manipulators, which are robots
# produced and sold by Hello Robot Inc.

# Copyright 2024 Hello Robot Inc.

# This software is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License
# v3.0 (GNU LGPLv3) as published by the Free Software Foundation.

# This software is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License v3.0 (GNU LGPLv3) for more details,
# which can be found via the following link:

# https://www.gnu.org/licenses/lgpl-3.0.en.html

# For further information about the Files including inquiries about
# dual licensing, please contact Hello Robot Inc.
###########################################################################

import errno
import math
import os
import time

import numpy as np
import urchin as urdf_loader

import stretch.motion.simple_ik_equations_numba as ie


def load_urdf(file_name):
    if not os.path.isfile(file_name):
        print()
        print("*****************************")
        print(
            "ERROR: "
            + file_name
            + " was not found. OptasIK requires a specialized URDF saved with this file name. prepare_base_rotation_ik_urdf.py can be used to generate this specialized URDF."
        )
        print("*****************************")
        print()
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), file_name)
    urdf = urdf_loader.URDF.load(file_name, lazy_load_meshes=True)
    return urdf


def get_joint_limits(urdf):
    joint_limits = {}
    for joint in urdf.actuated_joints:
        lower = float(joint.limit.lower)
        upper = float(joint.limit.upper)
        joint_limits[joint.name] = (lower, upper)
    return joint_limits


# Available with
# from hello_helpers import hello_misc
def angle_diff_rad(target_rad, current_rad):
    # I've written this type of function many times before, and it's
    # always been annoying and tricky. This time, I looked on the web:
    # https://stackoverflow.com/questions/1878907/the-smallest-difference-between-2-angles
    #
    # The output is restricted to be in the range -pi to pi.
    diff_rad = target_rad - current_rad
    diff_rad = ((diff_rad + math.pi) % (2.0 * math.pi)) - math.pi
    return diff_rad


def nan_in_configuration(configuration):
    for k, v in configuration.items():
        if math.isnan(v) or np.isnan(v):
            return True
    return False


class SimpleIK:
    def __init__(self):

        self.end_effector_name = "link_wrist_yaw"

        self.rotary_urdf_file_name = "./stretch_base_rotation_ik_with_fixed_wrist.urdf"
        self.prismatic_urdf_file_name = "./stretch_base_translation_ik_with_fixed_wrist.urdf"

        self.rotary_urdf = load_urdf(self.rotary_urdf_file_name)
        self.prismatic_urdf = load_urdf(self.prismatic_urdf_file_name)

        self.rotary_base_joints = [
            "joint_lift",
            "joint_arm_l0",
            "joint_mobile_base_rotation",
        ]
        self.prismatic_base_joints = [
            "joint_lift",
            "joint_arm_l0",
            "joint_mobile_base_translation",
        ]
        self.all_joints = [
            "joint_lift",
            "joint_arm_l0",
            "joint_mobile_base_rotation",
            "joint_mobile_base_translation",
        ]

        # Initialize with idealized kinematics

        # distance in the x direction between the mobile base rotational axis to the lift on the ground
        self.b1 = -0.1
        # unit vector in the direction of positive lift motion
        self.l_vec = np.array([0.0, 0.0, 1.0])
        # unit vector in the direction of positive arm extension
        self.a_vec = np.array([0.0, -1.0, 0.0])

        self.t_offset = 0.0
        self.m_offset = 0.0
        self.l_offset = 0.0
        self.a_offset = 0.0

        self.rotary_joint_limits = get_joint_limits(self.rotary_urdf)
        self.prismatic_joint_limits = get_joint_limits(self.prismatic_urdf)
        # Python version >= 3.9
        # self.all_joint_limits = self.rotary_joint_limits | self.prismatic_joint_limits
        self.all_joint_limits = {
            **self.rotary_joint_limits,
            **self.prismatic_joint_limits,
        }

        print()
        print("SimpleIK: self.rotary_joint_limits =", self.rotary_joint_limits)
        print("SimpleIK: self.prismatic_joint_limits =", self.prismatic_joint_limits)
        print()

        self.rotary_end_effector_link = self.rotary_urdf.link_map[self.end_effector_name]
        self.prismatic_end_effector_link = self.prismatic_urdf.link_map[self.end_effector_name]

        zero_cfg = {k: 0.0 for k in self.rotary_base_joints}
        zero_fk = self.rotary_urdf.link_fk(cfg=zero_cfg, links=[self.end_effector_name])
        zero_wrist_position = zero_fk[self.rotary_end_effector_link].dot(
            np.array([0.0, 0.0, 0.0, 1.0])
        )[:3]

        # find arm unit vector
        arm_cfg = zero_cfg.copy()
        arm_cfg["joint_arm_l0"] = 1.0
        arm_fk = self.rotary_urdf.link_fk(cfg=arm_cfg, links=[self.end_effector_name])
        arm_wrist_position = arm_fk[self.rotary_end_effector_link].dot(
            np.array([0.0, 0.0, 0.0, 1.0])
        )[:3]
        self.a_vec = arm_wrist_position - zero_wrist_position
        self.a_vec = self.a_vec / np.linalg.norm(self.a_vec)

        # find lift unit vector
        lift_cfg = zero_cfg.copy()
        lift_cfg["joint_lift"] = 1.0
        lift_fk = self.rotary_urdf.link_fk(cfg=lift_cfg, links=[self.end_effector_name])
        lift_wrist_position = lift_fk[self.rotary_end_effector_link].dot(
            np.array([0.0, 0.0, 0.0, 1.0])
        )[:3]
        self.l_vec = lift_wrist_position - zero_wrist_position
        self.l_vec = self.l_vec / np.linalg.norm(self.l_vec)

        x0, y0, z0 = zero_wrist_position
        a1, a2, a3 = self.a_vec
        l1, l2, l3 = self.l_vec

        mat = np.array([[a1, l1, 1.0], [a2, l2, 0.0], [a3, l3, 0.0]])
        inv = np.linalg.inv(mat)
        a_o, l_o, b1 = inv.dot(zero_wrist_position)

        self.a_offset = a_o
        self.l_offset = l_o
        # I've set b_2 to 0.0, because the lift and arm offsets create
        # an ambiguity. More specifically, they are redundant with a
        # two-dimensional vector for b
        self.b1 = b1

        self.force_numba_just_in_time_compilation()

    def within_joint_limits(self, robot_configuration):
        for joint_name in self.all_joints:
            joint_value = robot_configuration.get(joint_name, None)
            if joint_value is not None:
                lower_limit, upper_limit = self.all_joint_limits[joint_name]
                if joint_name == "joint_mobile_base_rotation":
                    # convert base angle to be in the range -pi to pi
                    # positive is to the robot's left side (clockwise)
                    # negative is to the robot's right side (counterclockwise)
                    joint_value = angle_diff_rad(joint_value, 0.0)
                if (joint_value < lower_limit) or (joint_value > upper_limit):
                    return False
        return True

    def clip_with_joint_limits(self, robot_configuration):
        # print('SimpleIK clip_with_joint_limits initial robot_configuration=', robot_configuration)
        for joint_name in self.all_joints:
            joint_value = robot_configuration.get(joint_name, None)
            if joint_value is not None:
                lower_limit, upper_limit = self.all_joint_limits[joint_name]
                if joint_name == "joint_mobile_base_rotation":
                    # convert base angle to be in the range -pi to pi
                    # positive is to the robot's left side (clockwise)
                    # negative is to the robot's right side (counterclockwise)
                    joint_value = angle_diff_rad(joint_value, 0.0)
                # print('SimpleIK clip_with_joint_limits initial joint_name, joint_value, lower_limit, upper_limit =',
                # joint_name, joint_value, lower_limit, upper_limit)
                robot_configuration[joint_name] = np.clip(joint_value, lower_limit, upper_limit)
                # print('SimpleIK clipped joint_value =', np.clip(joint_value, lower_limit, upper_limit))
        # print('SimpleIK clip_with_joint_limits final robot_configuration=', robot_configuration)

    def fk_rotary_base(self, robot_configuration, use_urdf=False):
        cfg = robot_configuration

        if use_urdf:
            urdf_fk = self.rotary_urdf.link_fk(cfg=cfg, links=[self.end_effector_name])
            wrist_position = urdf_fk[self.rotary_end_effector_link].dot(
                np.array([0.0, 0.0, 0.0, 1.0])
            )[:3]
        else:
            base_angle = cfg["joint_mobile_base_rotation"]
            lift_distance = cfg["joint_lift"] + self.l_offset
            arm_distance = cfg["joint_arm_l0"] + self.a_offset
            wrist_position = ie.calibrated_fk_with_rotary_base(
                base_angle=base_angle,
                lift_distance=lift_distance,
                arm_distance=arm_distance,
                b1=self.b1,
                l_vector=self.l_vec,
                a_vector=self.a_vec,
            )
        return wrist_position

    def ik_rotary_base(self, wrist_position):
        goal = np.array(wrist_position)
        T, L, A = ie.calibrated_ik_with_rotary_base(
            goal, b1=self.b1, l_vector=self.l_vec, a_vector=self.a_vec
        )
        cfg = {}
        cfg["joint_mobile_base_rotation"] = T
        cfg["joint_lift"] = L - self.l_offset
        cfg["joint_arm_l0"] = A - self.a_offset

        if nan_in_configuration(cfg) or (not self.within_joint_limits(cfg)):
            return None

        return cfg

    def force_numba_just_in_time_compilation(self):

        wrist_positions = [
            [0.1, -0.5, 0.3],
            [0.1, 0.5, 0.3],
            [-0.3, -0.5, 0.3],
            [0.2, -0.6, 0.6],
            [-0.4, -0.7, 0.3],
        ]

        for wrist_position_goal in wrist_positions:
            # Config is output here but not used.
            _ = self.ik_rotary_base(wrist_position_goal)

    def fk_prismatic_base(self, robot_configuration, use_urdf=False):
        cfg = robot_configuration

        if use_urdf:
            urdf_fk = self.prismatic_urdf.link_fk(cfg=cfg, links=[self.end_effector_name])
            wrist_position = urdf_fk[self.prismatic_end_effector_link].dot(
                np.array([0.0, 0.0, 0.0, 1.0])
            )[:3]
        else:
            base_distance = cfg["joint_mobile_base_translation"]
            lift_distance = cfg["joint_lift"] + self.l_offset
            arm_distance = cfg["joint_arm_l0"] + self.a_offset
            wrist_position = ie.calibrated_fk_with_prismatic_base(
                base_distance=base_distance,
                lift_distance=lift_distance,
                arm_distance=arm_distance,
                b1=self.b1,
                l_vector=self.l_vec,
                a_vector=self.a_vec,
            )
        return wrist_position

    def ik_prismatic_base(self, wrist_position):
        goal = np.array(wrist_position)
        M, L, A = ie.calibrated_ik_with_prismatic_base(
            goal, b1=self.b1, l_vector=self.l_vec, a_vector=self.a_vec
        )
        cfg = {}
        cfg["joint_mobile_base_translation"] = M
        cfg["joint_lift"] = L - self.l_offset
        cfg["joint_arm_l0"] = A - self.a_offset

        return cfg


if __name__ == "__main__":

    compare_with_optas_ik = False

    if compare_with_optas_ik:
        import optas_ik as oi

        optas_ik = oi.OptasIK(
            use_full_transform=False,
            use_fixed_wrist=True,
            visualize_ik=False,
            test_with_regular_urdf=False,
            debug_on=False,
        )
        nominal_ik_urdf_configuration = optas_ik.get_default_configuration()

    simple_ik = SimpleIK()

    wrist_positions = [
        [0.1, -0.5, 0.3],
        [0.1, 0.5, 0.3],
        [-0.3, -0.5, 0.3],
        [0.2, -0.6, 0.6],
        [-0.4, -0.7, 0.3],
    ]

    print()
    print("--- TEST SIMPLE FK FOR ROTARY BASE ---")

    robot_joint_values = [
        (0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 1.0, 1.0),
        (math.pi / 4.0, 0.0, 0.0),
        (math.pi / 4.0, 1.0, 0.0),
        (math.pi / 4.0, 0.0, 1.0),
        (math.pi / 4.0, 1.0, 1.0),
        (math.pi / 2.0, 0.0, 0.0),
        (math.pi / 2.0, 1.0, 0.0),
        (math.pi / 2.0, 0.0, 1.0),
        (math.pi / 2.0, 1.0, 1.0),
        (math.pi / 4.0, 0.0, 0.0),
        (math.pi / 4.0, 0.25, 0.0),
        (math.pi / 4.0, 0.0, 0.25),
        (math.pi / 4.0, 0.25, 0.25),
    ]

    cfg = {}

    for q in robot_joint_values:
        print()
        cfg["joint_mobile_base_rotation"] = q[0]
        cfg["joint_lift"] = q[1]
        cfg["joint_arm_l0"] = q[2]

        start_time = time.time()
        wrist_position_simple = simple_ik.fk_rotary_base(cfg, use_urdf=False)
        end_time = time.time()
        duration = end_time - start_time

        old_start_time = time.time()
        wrist_position_urdf = simple_ik.fk_rotary_base(cfg, use_urdf=True)
        old_end_time = time.time()
        old_duration = old_end_time - old_start_time

        print("joint configuration =", cfg)
        print("wrist_position_simple =", wrist_position_simple)
        print("wrist_position_urdf =", wrist_position_urdf)
        print("time for Simple FK =", "{:.4f}".format(duration * 1000.0), "milliseconds")
        speedup = old_duration / duration
        print("speedup over urchin FK =", "{:.4f}".format(speedup))
        if wrist_position_simple is not None:
            error = np.linalg.norm(wrist_position_urdf - wrist_position_simple)
            print("ERROR =", error)

    print()
    print("--- TEST SIMPLE IK FOR ROTARY BASE ---")

    for wrist_position_goal in wrist_positions:
        print()

        start_time = time.time()
        cfg = simple_ik.ik_rotary_base(wrist_position_goal)
        end_time = time.time()
        duration = end_time - start_time

        print("wrist_position_goal =", wrist_position_goal)
        if compare_with_optas_ik:
            old_start_time = time.time()
            (
                optimized_configuration,
                optimized_end_effector_pose,
            ) = optas_ik.perform_ik_optimization(wrist_position_goal, nominal_ik_urdf_configuration)
            old_end_time = time.time()
            old_duration = old_end_time - old_start_time
            print("Optas IK optimized configuration =", optimized_configuration)
            print(
                "Optas IK wrist position achieved with constraints =",
                optimized_end_effector_pose,
            )

        print("wrist_position_goal =", wrist_position_goal)
        print("IK configuration =", cfg)
        if cfg is not None:
            wrist_position_simple = simple_ik.fk_rotary_base(cfg, use_urdf=False)
            print("FK wrist position =", wrist_position_simple)

            simple_ik.clip_with_joint_limits(cfg)
            print("clipped IK configuration =", cfg)
            clipped_wrist_position_simple = simple_ik.fk_rotary_base(cfg, use_urdf=False)
            print("clipped FK wrist position =", clipped_wrist_position_simple)

            error = np.linalg.norm(np.array(wrist_position_goal) - wrist_position_simple)
            print(
                "time for Simple IK =",
                "{:.4f}".format(duration * 1000.0),
                "milliseconds",
            )
            print("ERROR =", error)
            if compare_with_optas_ik:
                speedup = old_duration / duration
                print("speedup over Optas IK =", "{:.4f}".format(speedup))

    print()
    print("--- TEST SIMPLE FK FOR PRISMATIC BASE ---")

    robot_joint_values = [
        (0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 1.0, 1.0),
        (0.5, 0.0, 0.0),
        (0.5, 1.0, 0.0),
        (0.5, 0.0, 1.0),
        (0.5, 1.0, 1.0),
        (-0.5, 0.0, 0.0),
        (-0.5, 1.0, 0.0),
        (-0.5, 0.0, 1.0),
        (-0.5, 1.0, 1.0),
    ]

    cfg = {}

    for q in robot_joint_values:
        print()
        cfg["joint_mobile_base_translation"] = q[0]
        cfg["joint_lift"] = q[1]
        cfg["joint_arm_l0"] = q[2]

        start_time = time.time()
        wrist_position_simple = simple_ik.fk_prismatic_base(cfg, use_urdf=False)
        end_time = time.time()
        duration = end_time - start_time

        old_start_time = time.time()
        wrist_position_urdf = simple_ik.fk_prismatic_base(cfg, use_urdf=True)
        old_end_time = time.time()
        old_duration = old_end_time - old_start_time

        print("joint configuration =", cfg)
        print("wrist_position_simple =", wrist_position_simple)
        print("wrist_position_urdf =", wrist_position_urdf)
        print("time for Simple FK =", "{:.4f}".format(duration * 1000.0), "milliseconds")
        speedup = old_duration / duration
        print("speedup over urchin FK =", "{:.4f}".format(speedup))
        if wrist_position_simple is not None:
            error = np.linalg.norm(wrist_position_urdf - wrist_position_simple)
            print("ERROR =", error)

    print()
    print("--- TEST SIMPLE IK FOR PRISMATIC BASE ---")

    for wrist_position_goal in wrist_positions:
        print()

        start_time = time.time()
        cfg = simple_ik.ik_prismatic_base(wrist_position_goal)
        end_time = time.time()
        duration = end_time - start_time

        print("wrist_position_goal =", wrist_position_goal)
        print("IK configuration =", cfg)
        if cfg is not None:
            wrist_position_simple = simple_ik.fk_prismatic_base(cfg, use_urdf=False)
            print("FK wrist position =", wrist_position_simple)

            simple_ik.clip_with_joint_limits(cfg)
            print("clipped IK configuration =", cfg)
            clipped_wrist_position_simple = simple_ik.fk_prismatic_base(cfg, use_urdf=False)
            print("clipped FK wrist position =", clipped_wrist_position_simple)

            error = np.linalg.norm(np.array(wrist_position_goal) - wrist_position_simple)
            print(
                "time for Simple IK =",
                "{:.4f}".format(duration * 1000.0),
                "milliseconds",
            )
            print("ERROR =", error)
