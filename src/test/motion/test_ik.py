# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import timeit
from typing import List

import numpy as np

from stretch.motion.pinocchio_ik_solver import PinocchioIKSolver

true_all_joints: List[str] = [
    "joint_mobile_base_rotation",
    "joint_lift",
    "joint_arm_l0",
    "joint_wrist_yaw",
    "joint_wrist_pitch",
    "joint_wrist_roll",
]


def test_fk_ik():
    # Create IK Solver
    urdf_path = os.path.dirname(__file__) + "/stretch_base_rotation_ik.urdf"
    print("URDF path =", urdf_path)
    ik_joints_allowed_to_move = [
        "joint_arm_l0",
        "joint_lift",
        "joint_wrist_yaw",
        "joint_wrist_pitch",
        "joint_wrist_roll",
        "joint_mobile_base_rotation",
    ]
    manip_ik_solver = PinocchioIKSolver(
        urdf_path,
        "link_grasp_center",
        ik_joints_allowed_to_move,
    )

    state = {
        "joint_mobile_base_rotation": 0.0,
        "joint_lift": 0.5,
        "joint_arm_l0": 0.5,
        "joint_wrist_pitch": 0,
        "joint_wrist_yaw": 0,
        "joint_wrist_roll": 0,
    }

    all_joints = manip_ik_solver.get_all_joint_names()
    for i, j in zip(all_joints, true_all_joints):
        assert i == j, f"Joint name mismatch: {i} != {j}"

    # Test Forward Kinematics
    ee_pose = manip_ik_solver.compute_fk(state)
    print(f"{ee_pose=}")
    assert ee_pose is not None, "FK failed"

    # Test Inverse Kinematics
    ee_position = np.array([-0.03, -0.4, 0.9])
    ee_orientation = np.array([0, 0, 0, 1])
    res, success, info = manip_ik_solver.compute_ik(
        ee_position,
        ee_orientation,
        q_init=state,
    )
    print("Result =", res)
    print("Success =", success)
    assert success, "IK failed"

    dt_sum = 0
    # Speed test
    for i in range(1000):
        # Test Inverse Kinematics
        t0 = timeit.default_timer()
        ee_position = np.random.rand(3) * 2 - 1
        ee_position[2] += 1
        ee_orientation = np.random.rand(4)
        ee_orientation /= np.linalg.norm(ee_orientation)
        res, success, info = manip_ik_solver.compute_ik(
            ee_position,
            ee_orientation,
            q_init=state,
        )
        t1 = timeit.default_timer()
        dt_sum += t1 - t0
    print("Average time per IK call =", dt_sum / 1000)
    hz = 1 / (dt_sum / 1000)
    print("Average rate =", hz)
    assert hz > 100, "IK solver too slow"


def test_ik_restricted():
    # Create IK Solver
    urdf_path = os.path.dirname(__file__) + "/stretch_base_rotation_ik.urdf"
    print("URDF path =", urdf_path)
    ik_joints_allowed_to_move = [
        "joint_arm_l0",
        "joint_lift",
        "joint_wrist_yaw",
        "joint_wrist_pitch",
        "joint_mobile_base_rotation",
    ]
    manip_ik_solver = PinocchioIKSolver(
        urdf_path,
        "link_grasp_center",
        ik_joints_allowed_to_move,
    )

    state = {
        "joint_mobile_base_rotation": 0.0,
        "joint_lift": 0.5,
        "joint_arm_l0": 0.5,
        "joint_wrist_pitch": 0,
        "joint_wrist_yaw": 0,
        # "joint_wrist_roll": 0,
    }

    all_joints = manip_ik_solver.get_all_joint_names()
    for i, j in zip(all_joints, true_all_joints):
        assert i == j, f"Joint name mismatch: {i} != {j}"

    # Test Forward Kinematics
    ee_pose = manip_ik_solver.compute_fk(state)
    print(f"{ee_pose=}")
    assert ee_pose is not None, "FK failed"

    # Test Inverse Kinematics
    ee_position = np.array([-0.03, -0.4, 0.9])
    ee_orientation = np.array([0, 0, 0, 1])
    res, success, info = manip_ik_solver.compute_ik(
        ee_position,
        ee_orientation,
        q_init=state,
    )
    print("Result =", res)
    print("Success =", success)
    assert success, "IK failed"


if __name__ == "__main__":
    test_fk_ik()
    test_ik_restricted()
