#!/usr/bin/env python3

# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import time
import timeit
from typing import Optional, Tuple, Union

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch

from stretch.mapping.scene_graph import SceneGraph
from stretch.mapping.voxel.voxel_map import SparseVoxelMapNavigationSpace
from stretch.motion import HelloStretchIdx
from stretch.perception.wrapper import OvmmPerception
from stretch.visualization import urdf_visualizer


def decompose_homogeneous_matrix(homogeneous_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decomposes a 4x4 homogeneous transformation matrix into its rotation matrix and translation vector components.

    Args:
        homogeneous_matrix (numpy.ndarray): A 4x4 matrix representing a homogeneous transformation.

    Returns:
        tuple: A tuple containing:
            - rotation_matrix : A 3x3 matrix representing the rotation component.
            - translation_vector : A 1D array of length 3 representing the translation component.
    """
    if homogeneous_matrix.shape != (4, 4):
        raise ValueError("Input matrix must be 4x4")
    rotation_matrix = homogeneous_matrix[:3, :3]
    translation_vector = homogeneous_matrix[:3, 3]
    return rotation_matrix, translation_vector


def occupancy_map_to_indices(occupancy_map):
    """
    Convert a 2D occupancy map to an Nx3 array of float indices of occupied cells.

    Args:
    occupancy_map (np.ndarray): 2D boolean array where True represents occupied cells.

    Returns:
    np.ndarray: Nx3 float array where each row is [x, y, 0] of an occupied cell.
    """
    # Find the indices of occupied cells
    occupied_indices = np.where(occupancy_map)

    # Create the Nx3 array
    num_points = len(occupied_indices[0])
    xyz_array = np.zeros((num_points, 3), dtype=float)

    # Fill in x and y coordinates
    xyz_array[:, 0] = occupied_indices[0]  # x coordinates
    xyz_array[:, 1] = occupied_indices[1]  # y coordinates
    # z coordinates are already 0

    return xyz_array


def occupancy_map_to_3d_points(
    occupancy_map: np.ndarray,
    grid_center: Union[np.ndarray, torch.Tensor],
    grid_resolution: float,
    offset: Optional[np.ndarray] = np.zeros(3),
) -> np.ndarray:
    """
    Converts a 2D occupancy map to a list of 3D points.
    Args:
        occupancy_map: A 2D array boolean map
        grid_center: The (x, y, z) coordinates of the center of the grid map
        grid_resolution: The resolution of the grid map
        offset: The (x, y, z) offset to be added to the points

    Returns:
        np.ndarray: A array of 3D points representing the occupied cells in the world frame.
    """
    points = []
    rows, cols = occupancy_map.shape
    center_row, center_col, _ = grid_center

    if isinstance(grid_center, torch.Tensor):
        grid_center = grid_center.cpu().numpy()

    indices = occupancy_map_to_indices(occupancy_map)
    points = (indices - grid_center) * grid_resolution + offset
    return points


class StretchURDFLogger(urdf_visualizer.URDFVisualizer):
    link_names = []
    link_poses = []

    def load_robot_mesh(self, cfg: dict = None, use_collision: bool = False):
        """
        Load robot mesh using urdf visualizer to rerun
        This is to be run once at the beginning of the rerun
        Args:
            cfg (dict): Configuration of the robot
            use_collision (bool): use collision mesh
        """
        trimesh_list = self.get_tri_meshes(cfg=cfg, use_collision=use_collision)
        self.link_names = trimesh_list["link"]
        self.link_poses = trimesh_list["pose"]
        for i in range(len(trimesh_list["link"])):
            rr.log(
                f"world/robot/mesh/{trimesh_list['link'][i]}",
                rr.Mesh3D(
                    vertex_positions=trimesh_list["mesh"][i].vertices,
                    triangle_indices=trimesh_list["mesh"][i].faces,
                    vertex_normals=trimesh_list["mesh"][i].vertex_normals,
                ),
                static=True,
            )

    def log_transforms(self, obs, debug: bool = False):
        """
        Log robot mesh using urdf visualizer to rerun
        Args:
            obs (dict): Observation dataclass
            use_collision (bool): use collision mesh
        """
        state = obs["joint"]
        cfg = {}
        for k in HelloStretchIdx.name_to_idx:
            cfg[k] = state[HelloStretchIdx.name_to_idx[k]]
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
        t0 = timeit.default_timer()
        tms = self.get_tri_meshes(cfg=lk_cfg, use_collision=False)
        t1 = timeit.default_timer()
        self.link_poses = tms["pose"]
        self.link_names = tms["link"]
        for link in self.link_names:
            idx = self.link_names.index(link)
            rr.set_time_seconds("realtime", time.time())
            rr.log(
                f"world/robot/mesh/{link}",
                rr.Transform3D(
                    translation=self.link_poses[idx][:3, 3],
                    mat3x3=self.link_poses[idx][:3, :3],
                    axis_length=0.0,
                ),
                static=False,
            )
        t2 = timeit.default_timer()
        if debug:
            print("Time to get tri meshes (ms): ", 1000 * (t1 - t0))
            print("Time to log robot transforms (ms): ", 1000 * (t2 - t1))
            print("Total time to log robot transforms (ms): ", 1000 * (t2 - t0))


class RerunVsualizer:
    def __init__(
        self,
        display_robot_mesh: bool = True,
        open_browser: bool = True,
        server_memory_limit: str = "4GB",
        collapse_panels: bool = True,
    ):
        """Rerun visualizer class
        Args:
            display_robot_mesh (bool): Display robot mesh
            open_browser (bool): Open browser at start
            server_memory_limit (str): Server memory limit E.g. 2GB or 20%
            collapse_panels (bool): Set to false to have customizable rerun panels
        """
        rr.init("Stretch_robot", spawn=False)
        rr.serve(open_browser=open_browser, server_memory_limit=server_memory_limit)

        self.display_robot_mesh = display_robot_mesh

        if self.display_robot_mesh:
            self.urdf_logger = StretchURDFLogger()
            self.urdf_logger.load_robot_mesh(use_collision=False)

        # Create environment Box place holder
        rr.log(
            "world/map_box",
            rr.Boxes3D(half_sizes=[10, 10, 3], centers=[0, 0, 2], colors=[255, 255, 255, 255]),
            static=True,
        )
        # World Origin
        rr.log(
            "world/xyz",
            rr.Arrows3D(
                vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            ),
            static=True,
        )

        self.bbox_colors_memory = {}
        self.step_delay_s = 0.3
        self.setup_blueprint(collapse_panels)

    def setup_blueprint(self, collapse_panels: bool):
        """Setup the blueprint for the visualizer
        Args:
            collapse_panels (bool): fully hides the blueprint/selection panels,
                                    and shows the simplified time panel
        """
        main = rrb.Horizontal(
            rrb.Spatial3DView(name="3D View", origin="world"),
            rrb.Vertical(
                rrb.Spatial2DView(name="head_rgb", origin="/world/head_camera"),
                rrb.Spatial2DView(name="ee_rgb", origin="/world/ee_camera"),
            ),
            column_shares=[3, 1],
        )
        my_blueprint = rrb.Blueprint(
            rrb.Vertical(main, rrb.TimePanel(state=True)),
            collapse_panels=collapse_panels,
        )
        rr.send_blueprint(my_blueprint)

    def log_head_camera(self, obs):
        """Log head camera pose and images"""
        rr.set_time_seconds("realtime", time.time())
        rr.log("world/head_camera/rgb", rr.Image(obs["rgb"]), static=True)
        rr.log("world/head_camera/depth", rr.DepthImage(obs["depth"]), static=True)
        rot, trans = decompose_homogeneous_matrix(obs["camera_pose"])
        rr.log(
            "world/head_camera",
            rr.Transform3D(translation=trans, mat3x3=rot, axis_length=0.3),
            static=True,
        )
        rr.log(
            "world/head_camera",
            rr.Pinhole(
                resolution=[obs["rgb"].shape[1], obs["rgb"].shape[0]],
                image_from_camera=obs["camera_K"],
                image_plane_distance=0.15,
            ),
            static=True,
        )

    def log_robot_xyt(self, obs):
        """Log robot world pose"""
        rr.set_time_seconds("realtime", time.time())
        xy = obs["gps"]
        theta = obs["compass"]
        rb_arrow = rr.Arrows3D(
            origins=[0, 0, 0],
            vectors=[0.4, 0, 0],
            radii=0.02,
            labels="robot",
            colors=[255, 0, 0, 255],
        )
        rr.log("world/robot/arrow", rb_arrow, static=True)
        rr.log(
            "world/robot/blob",
            rr.Points3D([0, 0, 0], colors=[255, 0, 0, 255], radii=0.13),
            static=True,
        )
        rr.log(
            "world/robot",
            rr.Transform3D(
                translation=[xy[0], xy[1], 0],
                rotation=rr.RotationAxisAngle(axis=[0, 0, 1], radians=theta),
                axis_length=0.7,
            ),
            static=True,
        )

    def log_ee_frame(self, obs):
        """log end effector pose
        Args:
            obs (Observations): Observation dataclass
        """
        rr.set_time_seconds("realtime", time.time())
        # EE Frame
        rot, trans = decompose_homogeneous_matrix(obs["ee_pose"])
        ee_arrow = rr.Arrows3D(
            origins=[0, 0, 0], vectors=[0.2, 0, 0], radii=0.02, labels="ee", colors=[0, 255, 0, 255]
        )
        rr.log("world/ee/arrow", ee_arrow, static=True)
        rr.log(
            "world/ee", rr.Transform3D(translation=trans, mat3x3=rot, axis_length=0.3), static=True
        )

    def log_ee_camera(self, servo):
        """Log end effector camera pose and images
        Args:
            servo (Servo): Servo observation dataclass
        """
        rr.set_time_seconds("realtime", time.time())
        # EE Camera
        rr.log("world/ee_camera/rgb", rr.Image(servo.ee_rgb), static=True)
        rr.log("world/ee_camera/depth", rr.DepthImage(servo.ee_depth), static=True)
        rot, trans = decompose_homogeneous_matrix(servo.ee_camera_pose)
        rr.log(
            "world/ee_camera",
            rr.Transform3D(translation=trans, mat3x3=rot, axis_length=0.3),
            static=True,
        )
        rr.log(
            "world/ee_camera",
            rr.Pinhole(
                resolution=[servo.ee_rgb.shape[1], servo.ee_rgb.shape[0]],
                image_from_camera=servo.ee_camera_K,
                image_plane_distance=0.15,
            ),
            static=True,
        )

    def log_robot_state(self, obs):
        """Log robot joint states"""
        rr.set_time_seconds("realtime", time.time())
        state = obs["joint"]
        for k in HelloStretchIdx.name_to_idx:
            rr.log(
                f"robot_state/joint_pose/{k}",
                rr.Scalar(state[HelloStretchIdx.name_to_idx[k]]),
                static=True,
            )

    def log_robot_transforms(self, obs):
        """
        Log robot mesh transforms using urdf visualizer"""
        self.urdf_logger.log_transforms(obs)

    def update_voxel_map(
        self,
        space: SparseVoxelMapNavigationSpace,
        debug: bool = False,
        explored_radius=0.01,
        obstacle_radius=0.05,
    ):
        """Log voxel map and send it to Rerun visualizer
        Args:
            space (SparseVoxelMapNavigationSpace): Voxel map object
        """
        rr.set_time_seconds("realtime", time.time())

        t0 = timeit.default_timer()
        points, _, _, rgb = space.voxel_map.voxel_pcd.get_pointcloud()
        if rgb is None:
            return

        rr.log(
            "world/point_cloud",
            rr.Points3D(positions=points, radii=np.ones(rgb.shape[0]) * 0.01, colors=np.int64(rgb)),
            static=True,
        )

        t1 = timeit.default_timer()
        grid_origin = space.voxel_map.grid_origin
        t2 = timeit.default_timer()
        obstacles, explored = space.voxel_map.get_2d_map()
        t3 = timeit.default_timer()

        # Get obstacles and explored points
        grid_resolution = space.voxel_map.grid_resolution
        obs_points = np.array(occupancy_map_to_3d_points(obstacles, grid_origin, grid_resolution))
        t4 = timeit.default_timer()

        # Get explored points
        explored_points = np.array(
            occupancy_map_to_3d_points(explored, grid_origin, grid_resolution)
        )
        t5 = timeit.default_timer()

        # Log points
        rr.log(
            "world/obstacles",
            rr.Points3D(
                positions=obs_points,
                radii=np.ones(points.shape[0]) * obstacle_radius,
                colors=[255, 0, 0],
            ),
            static=True,
        )
        rr.log(
            "world/explored",
            rr.Points3D(
                positions=explored_points,
                radii=np.ones(points.shape[0]) * explored_radius,
                colors=[255, 255, 255],
            ),
            static=True,
        )
        t6 = timeit.default_timer()

        if debug:
            print("Time to get point cloud: ", t1 - t0, "% = ", (t1 - t0) / (t6 - t0))
            print("Time to get grid origin: ", t2 - t1, "% = ", (t2 - t1) / (t6 - t0))
            print("Time to get 2D map: ", t3 - t2, "% = ", (t3 - t2) / (t6 - t0))
            print("Time to get obstacles points: ", t4 - t3, "% = ", (t4 - t3) / (t6 - t0))
            print("Time to get explored points: ", t5 - t4, "% = ", (t5 - t4) / (t6 - t0))
            print("Time to log points: ", t6 - t5, "% = ", (t6 - t5) / (t6 - t0))

    def update_scene_graph(
        self, scene_graph: SceneGraph, semantic_sensor: Optional[OvmmPerception] = None
    ):
        """Log objects bounding boxes and relationships
        Args:
            scene_graph (SceneGraph): Scene graph object
            semantic_sensor (OvmmPerception): Semantic sensor object
        """
        if semantic_sensor:
            rr.set_time_seconds("realtime", time.time())
            centers = []
            labels = []
            bounds = []
            colors = []

            t0 = timeit.default_timer()
            for idx, instance in enumerate(scene_graph.instances):
                name = semantic_sensor.get_class_name_for_id(instance.category_id)
                if name not in self.bbox_colors_memory:
                    self.bbox_colors_memory[name] = np.random.randint(0, 255, 3)
                best_view = instance.get_best_view()
                bbox_bounds = best_view.bounds  # 3D Bounds
                point_cloud_rgb = instance.point_cloud
                pcd_rgb = instance.point_cloud_rgb
                rr.log(
                    f"world/{instance.id}_{name}",
                    rr.Points3D(positions=point_cloud_rgb, colors=np.int64(pcd_rgb)),
                    static=True,
                )
                half_sizes = [(b[0] - b[1]) / 2 for b in bbox_bounds]
                bounds.append(half_sizes)
                pose = scene_graph.get_ins_center_pos(idx)
                confidence = best_view.score
                centers.append(rr.components.PoseTranslation3D(pose))
                labels.append(f"{name} {confidence:.2f}")
                colors.append(self.bbox_colors_memory[name])
            rr.log(
                "world/objects",
                rr.Boxes3D(
                    half_sizes=bounds,
                    centers=centers,
                    labels=labels,
                    radii=0.01,
                    colors=colors,
                ),
                static=True,
            )
            t1 = timeit.default_timer()
            print("Time to log scene graph objects: ", t1 - t0)

    def update_nav_goal(self, goal, timeout=10):
        """Log navigation goal
        Args:
            goal (np.ndarray): Goal coordinates
        """
        ts = time.time()
        rr.set_time_seconds("realtime", ts)
        rr.log("world/xyt_goal", rr.Points3D([0, 0, 0], colors=[0, 255, 0, 50], radii=0.1))
        rr.log(
            "world/xyt_goal",
            rr.Transform3D(
                translation=[goal[0], goal[1], 0],
                rotation=rr.RotationAxisAngle(axis=[0, 0, 1], radians=goal[2]),
                axis_length=0.5,
            ),
        )
        # rr.set_time_seconds("realtime", ts + timeout)
        # rr.log("world/xyt_goal", rr.Clear(recursive=True))
        # rr.set_time_seconds("realtime", ts)

    def step(self, obs, servo):
        """Log all the data"""
        if obs and servo:
            rr.set_time_seconds("realtime", time.time())
            try:
                t0 = timeit.default_timer()
                self.log_robot_xyt(obs)
                self.log_head_camera(obs)
                self.log_ee_frame(obs)
                self.log_ee_camera(servo)
                self.log_robot_state(obs)

                if self.display_robot_mesh:
                    self.log_robot_transforms(obs)
                t1 = timeit.default_timer()
                sleep_time = self.step_delay_s - (t1 - t0)
                if sleep_time > 0:
                    time.sleep(sleep_time)

            except Exception as e:
                print(e)
