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
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pinocchio
from scipy.spatial.transform import Rotation as R

import stretch.utils.logger as logger
from stretch.motion.base.ik_solver_base import IKSolverBase

# --DEFAULTS--
# Error tolerances
POS_ERROR_TOL = 0.005
ORI_ERROR_TOL = [0.1, 0.1, np.pi / 2]

# CEM
CEM_MAX_ITERATIONS = 5
CEM_NUM_SAMPLES = 50
CEM_NUM_TOP = 10


def level_filter(levels):
    """Filter log messages by level"""

    def is_level(record):
        return record["level"].name in levels

    return is_level


class PinocchioIKSolver(IKSolverBase):
    """IK solver using pinocchio which can handle end-effector constraints for optimized IK solutions"""

    EPS = 1e-4
    DT = 1e-1
    DAMP = 1e-12

    def __init__(
        self, urdf_path: str, ee_link_name: str, controlled_joints: List[str], verbose: bool = False
    ):
        """
        urdf_path: path to urdf file
        ee_link_name: name of the end-effector link
        controlled_joints: list of joint names to control
        """
        if verbose:
            print(f"{urdf_path=}")
        self.model = pinocchio.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.q_neutral = pinocchio.neutral(self.model)

        # print([f.name for f in self.model.frames])
        self.ee_frame_idx = [f.name for f in self.model.frames].index(ee_link_name)

        self.controlled_joints_by_name = {}
        self.controlled_joints = []
        self.controlled_joint_names = controlled_joints
        for joint in controlled_joints:
            if joint == "ignore":
                idx = -1
            else:
                jid = self.model.getJointId(joint)
                if jid >= len(self.model.idx_qs):
                    logger.error(f"{joint=} {jid=} not in model.idx_qs")
                    raise RuntimeError(f"Invalid urdf at {urdf_path=}: missing {joint=}")
                else:
                    idx = self.model.idx_qs[jid]
            self.controlled_joints.append(idx)
            self.controlled_joints_by_name[joint] = idx

        logger.info(f"{controlled_joints=}")
        for j in controlled_joints:
            idx = self.model.getJointId(j)
            idx_q = self.model.idx_qs[idx]
            logger.info(f"{j=} {idx=} {idx_q=}")

    def get_dof(self) -> int:
        """returns dof for the manipulation chain"""
        return len(self.controlled_joints)

    def get_num_controllable_joints(self) -> int:
        """returns number of controllable joints under this solver's purview"""
        return len(self.controlled_joints)

    def get_all_joint_names(self) -> List[str]:
        """Return a list of joints"""
        return [self.model.names[i + 1] for i in range(self.model.nq)]

    def _qmap_control2model(
        self, q_input: Union[np.ndarray, dict], ignore_missing_joints: bool = False
    ) -> np.ndarray:
        """returns a full joint configuration from a partial joint configuration"""
        q_out = self.q_neutral.copy()
        if isinstance(q_input, dict):
            for joint_name, value in q_input.items():
                if joint_name in self.controlled_joints_by_name:
                    q_out[self.controlled_joints_by_name[joint_name]] = value
                else:
                    jid = self.model.getJointId(joint_name)
                    if jid >= len(self.model.idx_qs):
                        if not ignore_missing_joints:
                            logger.error(f"ERROR: {joint_name=} {jid=} not in model.idx_qs")
                            raise RuntimeError(
                                f"Tried to set joint not in model.idx_qs: {joint_name=}"
                            )
                    else:
                        q_out[self.model.idx_qs[self.model.getJointId(joint_name)]] = value
        else:
            assert len(self.controlled_joints) == len(
                q_input
            ), "if not specifying by name, must match length"
            for i, joint_idx in enumerate(self.controlled_joints):
                q_out[joint_idx] = q_input[i]
        return q_out

    def _qmap_model2control(self, q_input: np.ndarray) -> np.ndarray:
        """returns a partial joint configuration from a full joint configuration"""
        q_out = np.empty(len(self.controlled_joints))
        for i, joint_idx in enumerate(self.controlled_joints):
            if joint_idx >= 0:
                q_out[i] = q_input[joint_idx]

        return q_out

    def get_frame_pose(
        self,
        config: Union[np.ndarray, dict],
        node_a: str,
        node_b: str,
        ignore_missing_joints: bool = False,
    ):
        """
        Get a transformation matrix transforming from node_a frame to node_b frame
        """
        q_model = self._qmap_control2model(config, ignore_missing_joints=ignore_missing_joints)
        # print('q_model', q_model)
        pinocchio.forwardKinematics(self.model, self.data, q_model)
        frame_idx1 = [f.name for f in self.model.frames].index(node_a)
        frame_idx2 = [f.name for f in self.model.frames].index(node_b)
        # print(frame_idx1)
        # print(frame_idx2)
        # print(self.model.getFrameId(node_a))
        # print(self.model.getFrameId(node_b))
        # frame_idx1 = self.model.getFrameId(node_a)
        # frame_idx2 = self.model.getFrameId(node_b)
        pinocchio.updateFramePlacement(self.model, self.data, frame_idx1)
        placement_frame1 = self.data.oMf[frame_idx1]
        pinocchio.updateFramePlacement(self.model, self.data, frame_idx2)
        placement_frame2 = self.data.oMf[frame_idx2]
        # print('pin 1', placement_frame1)
        # print('pin 2', placement_frame2)
        return placement_frame2.inverse() * placement_frame1

    def compute_fk(
        self, config: np.ndarray, link_name: str = None, ignore_missing_joints: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Given joint values, return end-effector position and quaternion associated with it.

        Args:
            config: joint values
            link_name: name of the link to compute FK for; if None, uses the end-effector link

        Returns:
            pos: end-effector position (x, y, z)
            quat: end-effector quaternion (w, x, y, z)
        """
        if link_name is None:
            frame_idx = self.ee_frame_idx
        else:
            try:
                frame_idx = [f.name for f in self.model.frames].index(link_name)
            except ValueError:
                logger.error(f"Unknown link_name {link_name}. Defaulting to end-effector")
                frame_idx = self.ee_frame_idx
        q_model = self._qmap_control2model(config, ignore_missing_joints=ignore_missing_joints)
        pinocchio.forwardKinematics(self.model, self.data, q_model)
        pinocchio.updateFramePlacement(self.model, self.data, frame_idx)
        pos = self.data.oMf[frame_idx].translation
        quat = R.from_matrix(self.data.oMf[frame_idx].rotation).as_quat()
        return pos.copy(), quat.copy()

    def compute_ik(
        self,
        pos_desired: np.ndarray,
        quat_desired: np.ndarray,
        q_init=None,
        max_iterations=100,
        num_attempts: int = 1,
        verbose: bool = False,
        ignore_missing_joints: bool = False,
        node_name=None,
    ) -> Tuple[np.ndarray, bool, dict]:
        """given end-effector position and quaternion, return joint values.

        Two parameters are currently unused and might be implemented in the future:
            q_init: initial configuration for the optimization to start in; especially useful for
                    arms with redundant degrees of freedom
            num_attempts: start from multiple initial configs; included for compatibility with pb
            max iterations: time budget in number of steps; included for compatibility with pb
        """
        i = 0

        if q_init is None:
            q = self.q_neutral.copy()
            if num_attempts > 1:
                raise NotImplementedError(
                    "Sampling multiple initial configs not yet supported by Pinocchio solver."
                )
        else:
            q = self._qmap_control2model(q_init, ignore_missing_joints=ignore_missing_joints)
            # Override the number of attempts
            num_attempts = 1

        desired_ee_pose = pinocchio.SE3(R.from_quat(quat_desired).as_matrix(), pos_desired)
        while True:
            pinocchio.forwardKinematics(self.model, self.data, q)
            if node_name is not None:
                frame_idx = [f.name for f in self.model.frames].index(node_name)
            else:
                frame_idx = self.ee_frame_idx
            pinocchio.updateFramePlacement(self.model, self.data, frame_idx)

            dMi = desired_ee_pose.actInv(self.data.oMf[frame_idx])
            err = pinocchio.log(dMi).vector
            if verbose:
                print(f"[pinocchio_ik_solver] iter={i}; error={err}")
            if np.linalg.norm(err) < self.EPS:
                success = True
                break
            if i >= max_iterations:
                success = False
                break
            J = pinocchio.computeFrameJacobian(
                self.model,
                self.data,
                q,
                frame_idx,
                pinocchio.ReferenceFrame.LOCAL,
            )
            v = -J.T.dot(np.linalg.solve(J.dot(J.T) + self.DAMP * np.eye(6), err))
            q = pinocchio.integrate(self.model, q, v * self.DT)
            i += 1

        q_control = self._qmap_model2control(q.flatten())
        debug_info = {"iter": i, "final_error": err}

        return q_control, success, debug_info

    def q_array_to_dict(self, arr: np.ndarray):
        state = {}
        assert len(arr) == len(self.controlled_joint_names)
        for i, name in enumerate(self.controlled_joint_names):
            state[name] = arr[i]
        return state


class PositionIKOptimizer(IKSolverBase):
    """
    Solver that jointly optimizes IK and best orientation to achieve desired position.
    Can optimize any solver that implements IKSolverBase.
    Additionally, it implements IKSolverBase so this optimizer-based version can be readily dropped-in.
    """

    max_iterations: int = 30  # Max num of iterations for CEM
    num_samples: int = 100  # Total candidate samples for each CEM iteration
    num_top: int = 10  # Top N candidates for each CEM iteration

    def __init__(
        self,
        ik_solver: IKSolverBase,
        pos_error_tol: float,
        ori_error_range: Union[float, np.ndarray],
        pos_weight: float = 1.0,
        ori_weight: float = 0.0,
        cem_params: Optional[Dict] = None,
    ):
        self.pos_wt = pos_weight
        self.ori_wt = ori_weight

        # Initialize IK solver
        self.ik_solver = ik_solver

        # Initialize optimizer
        self.pos_error_tol = pos_error_tol
        if type(ori_error_range) is float:
            self.ori_error_range = ori_error_range * np.ones(3)
        else:
            self.ori_error_range = ori_error_range  # type: ignore

        cem_params = {} if cem_params is None else cem_params
        max_iterations = (
            cem_params["max_iterations"] if "max_iterations" in cem_params else self.max_iterations
        )
        num_samples = cem_params["num_samples"] if "num_samples" in cem_params else self.num_samples
        num_top = cem_params["num_top"] if "num_top" in cem_params else self.num_top

        self.opt = CEM(
            max_iterations=max_iterations,
            num_samples=num_samples,
            num_top=num_top,
            tol=self.pos_error_tol,
            sigma0=self.ori_error_range / 2,
        )

    def get_dof(self) -> int:
        return self.ik_solver.get_dof()

    def get_num_controllable_joints(self) -> int:
        return self.ik_solver.get_num_controllable_joints()

    def compute_ik(
        self,
        pos_desired: np.ndarray,
        quat_desired: np.ndarray,
        *args,
        **kwargs,
    ) -> Tuple[np.ndarray, bool, dict]:
        """optimization-based IK solver using CEM"""

        # Function to optimize: IK error given delta from original desired orientation
        def solve_ik(dr):
            pos = pos_desired
            quat = (R.from_rotvec(dr) * R.from_quat(quat_desired)).as_quat()

            q, _, subsolver_debug_info = self.ik_solver.compute_ik(pos, quat)
            pos_out, rot_out = self.ik_solver.compute_fk(q)

            cost_pos = np.linalg.norm(pos - pos_out)
            cost_rot = 1 - (rot_out * quat_desired).sum() ** 2  # TODO: just minimize dr?

            cost = self.pos_wt * cost_pos + self.ori_wt * cost_rot

            return cost, q

        # Optimize for IK and best orientation (x=0 -> use original desired orientation)
        cost_opt, q_result, max_iter, opt_sigma, success = self.opt.optimize(
            solve_ik, x0=np.zeros(3)
        )
        pos_out, quat_out = self.ik_solver.compute_fk(q_result)
        print(
            f"After ik optimization, cost: {cost_opt}, result: {pos_out, quat_out} vs desired: {pos_desired, quat_desired}"
        )

        debug_info = {
            "best_cost": cost_opt,
            "last_iter": max_iter,
            "opt_sigma": opt_sigma,
        }

        return q_result, success, debug_info

    def compute_fk(self, q):
        return self.ik_solver.compute_fk(q)


class CEM:
    """class implementing generic CEM solver for optimization"""

    def __init__(
        self,
        max_iterations: int,
        num_samples: int,
        num_top: int,
        tol: float,
        sigma0: np.ndarray,
    ):
        """
        max_iterations: max number of iterations
        num_samples: number of samples per iteration
        num_top: number of top samples to use for next iteration
        tol: tolerance for stopping criterion
        """
        self.max_iterations = max_iterations
        self.num_samples = num_samples
        self.num_top = num_top
        self.cost_tol = tol
        self.sigma0 = sigma0

    def optimize(self, func: Callable, x0: np.ndarray):
        """optimize function func with initial guess mu=x0 and initial std=sigma0"""
        assert (
            x0.shape == self.sigma0.shape
        ), f"x0 and sigma0 must have same shape, got {x0.shape} and {self.sigma0.shape}"

        i = 0
        mu = x0
        sigma = self.sigma0

        while True:
            # Sample x
            x_arr = mu + sigma * np.random.randn(self.num_samples, x0.shape[0])

            # Compute costs
            cost_arr = np.zeros(self.num_samples)
            aux_outputs = [None for _ in range(self.num_samples)]
            for j, x in enumerate(x_arr):
                cost_arr[j], aux_outputs[j] = func(x)

            # Sort costs
            idx_sorted_arr = np.argsort(cost_arr)
            i_best = idx_sorted_arr[0]

            # Check termination
            i += 1
            if i >= self.max_iterations or np.all(sigma <= self.cost_tol / 10):
                # If we have run out of iterations or if our sigma has converged before getting close enough, per our
                # error tolerances, then the optimization failed
                success = False
                break

            if cost_arr[i_best] <= self.cost_tol:
                success = True
                break

            # Update distribution
            mu = np.mean(x_arr[idx_sorted_arr[: self.num_top], :], axis=0)
            sigma = np.std(x_arr[idx_sorted_arr[: self.num_top], :], axis=0)

        return cost_arr[i_best], aux_outputs[i_best], i, sigma, success
