# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import timeit
from typing import List

import numpy as np

from stretch.motion.pinocchio_ik_solver import PinocchioIKSolver

EPS_IK_CORRECT = 0.05  # max IK error

true_all_joints_revolute: List[str] = [
    "joint_mobile_base_rotation",
    "joint_lift",
    "joint_arm_l0",
    "joint_wrist_yaw",
    "joint_wrist_pitch",
    "joint_wrist_roll",
]

true_all_joints_translation: List[str] = [
    "joint_mobile_base_translation",
    "joint_lift",
    "joint_arm_l0",
    "joint_wrist_yaw",
    "joint_wrist_pitch",
    "joint_wrist_roll",
]

state_revolute = {
    "joint_mobile_base_rotation": 0.0,
    "joint_lift": 0.5,
    "joint_arm_l0": 0.5,
    "joint_wrist_pitch": 0,
    "joint_wrist_yaw": 0,
    "joint_wrist_roll": 0,
}

state_translation = {
    "joint_mobile_base_translation": 0.0,
    "joint_lift": 0.5,
    "joint_arm_l0": 0.5,
    "joint_wrist_pitch": 0,
    "joint_wrist_yaw": 0,
    "joint_wrist_roll": 0,
}


def test_fk_ik():
    _test_fk_ik("stretch_base_rotation_ik.urdf", true_all_joints_revolute, state_revolute)
    _test_fk_ik("stretch_base_translation_ik.urdf", true_all_joints_translation, state_translation)


def _test_fk_ik(urdf_file, true_joint_names, initial_joint_state):
    # Create IK Solver
    try:
        urdf_path = os.path.join(os.path.dirname(__file__), urdf_file)
    except FileNotFoundError as e:
        print(e)
        assert False, "URDF file not found!"

    print("URDF path =", urdf_path)
    ik_joints_allowed_to_move = initial_joint_state.keys()

    manip_ik_solver = PinocchioIKSolver(
        urdf_path,
        "link_grasp_center",
        ik_joints_allowed_to_move,
    )

    all_joints = manip_ik_solver.get_all_joint_names()
    for i, j in zip(all_joints, true_joint_names):
        assert i == j, f"Joint name mismatch: {i} != {j}"

    # Test Forward Kinematics
    ee_pose = manip_ik_solver.compute_fk(initial_joint_state)
    print(f"{ee_pose=}")
    assert ee_pose is not None, "FK failed"

    # Test Inverse Kinematics
    ee_position = np.array([-0.03, -0.4, 0.9])
    ee_orientation = np.array([0, 0, 0, 1])
    res, success, info = manip_ik_solver.compute_ik(
        ee_position,
        ee_orientation,
        q_init=initial_joint_state,
    )
    print("Result =", res)
    print("Success =", success)
    assert success, "IK failed"

    # Test IK accuracy
    res_ee_position, res_ee_orientation = manip_ik_solver.compute_fk(res)
    ee_position_error = np.linalg.norm(res_ee_position - ee_position)
    ee_orientation_error = np.linalg.norm(res_ee_orientation - ee_orientation)
    assert ee_position_error < EPS_IK_CORRECT, "IK position error too large"
    assert ee_orientation_error < EPS_IK_CORRECT, "IK orientation error too large"

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
            q_init=initial_joint_state,
        )
        t1 = timeit.default_timer()
        dt_sum += t1 - t0
    print("Average time per IK call =", dt_sum / 1000)
    hz = 1 / (dt_sum / 1000)
    print("Average rate =", hz)
    assert hz > 100, "IK solver too slow"


def test_fk_various_links():
    """From Amal:
    here are some FKs that I know are correct. (By "know are correct," I mean I use them in click-to-pregrasp and they work, and that I've roughly measured the distances and they align with what I'd expect)
    base_link --> link_lift:
        q:
        joint_mobile_base_rotation: 0.
        joint_lift: 0.60068669
        joint_arm: 0.10000367
        wrist_yaw: 0.00830906
        wrist_pitch: -0.64273795
        wrist_roll: -0.00306796
        pos: [-0.11694921  0.15266815  0.79586019]
        quat: [-0.01168736  0.00655014  0.70903008  0.70505095]
    base_link --> link_arm_l0:
        q:
        joint_mobile_base_rotation: 0.
        joint_lift: 0.60068669
        joint_arm: 0.10000367
        wrist_yaw: 0.0
        wrist_pitch: -1.57079633
        wrist_roll: 0.0
        pos: [-0.10633073 -0.23019249  0.79947608]
        quat: [0.69547437 -0.00870865  0.02678533  0.71799868]
    base_link --> link_grasp_center:
        q:
        joint_mobile_base_rotation: 0.
        joint_lift: 0.60068669
        joint_arm: 0.10000367
        wrist_yaw: 0.0
        wrist_pitch: -1.57079633
        wrist_roll: 0.0
        pos: [-0.00721532 -0.21435387  0.47230771]
        quat: [0.49792886  0.48561655 -0.48876169  0.52664544]
    """
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

    test_states = [
        {
            "state": {
                "joint_mobile_base_rotation": 0.0,
                "joint_lift": 0.60068669,
                "joint_arm_l0": 0.10000367,
                "joint_wrist_yaw": 0.00830906,
                "joint_wrist_pitch": -0.64273795,
                "joint_wrist_roll": -0.00306796,
            },
            "link_name": "link_lift",
            "expected_pos": [-0.11694921, 0.15266815, 0.79586019],
            "expected_quat": [-0.01168736, 0.00655014, 0.70903008, 0.70505095],
        },
        {
            "state": {
                "joint_mobile_base_rotation": 0.0,
                "joint_lift": 0.60068669,
                "joint_arm_l0": 0.10000367,
                "joint_wrist_yaw": 0.0,
                "joint_wrist_pitch": -1.57079633,
                "joint_wrist_roll": 0.0,
            },
            "expected_pos": [-0.10633073, -0.23019249, 0.79947608],
            "expected_quat": [0.69547437, -0.00870865, 0.02678533, 0.71799868],
            "link_name": "link_arm_l0",
        },
        {
            "state": {
                "joint_mobile_base_rotation": 0.0,
                "joint_lift": 0.60068669,
                "joint_arm_l0": 0.10000367,
                "joint_wrist_yaw": 0.0,
                "joint_wrist_pitch": -1.57079633,
                "joint_wrist_roll": 0.0,
            },
            "expected_pos": [-0.00721532, -0.21435387, 0.47230771],
            "expected_quat": [0.49792886, 0.48561655, -0.48876169, 0.52664544],
            "link_name": "link_grasp_center",
        },
    ]

    # Run the tests
    for i, test in enumerate(test_states):
        state = test["state"]
        expected_pos = np.array(test["expected_pos"])
        expected_quat = np.array(test["expected_quat"])
        link = test["link_name"]

        pos, quat = manip_ik_solver.compute_fk(state, link_name=link)

        pos_diff = np.linalg.norm(expected_pos - pos)
        quat_diff = np.linalg.norm(expected_quat - quat)

        print(f"Forward Kinematics Test {i} for {link=}:")
        print(f"Position difference: {pos_diff}")
        print(f"Quaternion difference: {quat_diff}")
        print()

        assert pos_diff < EPS_IK_CORRECT, "Position difference too large"
        assert quat_diff < EPS_IK_CORRECT, "Quaternion difference too large"


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
    for i, j in zip(all_joints, true_all_joints_revolute):
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
    test_fk_various_links()
