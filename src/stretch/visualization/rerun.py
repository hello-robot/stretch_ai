#!/usr/bin/env python3

import time
from typing import Optional, Tuple

import numpy as np
import rerun as rr


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


def occupancy_map_to_3d_points(
    occupancy_map: np.ndarray,
    grid_center: np.ndarray,
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

    for i in range(rows):
        for j in range(cols):
            if occupancy_map[i][j]:
                x = (i - center_col) * grid_resolution
                y = (j - center_col) * grid_resolution
                z = 0  # Assuming the map is 2D, so z is always 0
                x = x + offset[0]
                y = y + offset[1]
                z = z + offset[2]
                points.append(np.array([x, y, z]))

    return np.array(points)


class RerunVsualizer:
    def __init__(self):
        rr.init("Stretch_robot", spawn=False)
        rr.serve(open_browser=False, server_memory_limit="1GB")

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
        )

        self.bbox_colors_memory = {}

    def log_head_camera(self, obs):
        """Log head camera pose and images"""
        rr.set_time_seconds('realtime', time.time())
        rr.log("world/head_camera/rgb", rr.Image(obs["rgb"]))
        rr.log("world/head_camera/depth", rr.DepthImage(obs["depth"], meter=1000.0))
        rr.log(
            "world/head_camera/rgb",
            rr.Pinhole(
                resolution=[obs["rgb"].shape[1], obs["rgb"].shape[0]],
                image_from_camera=obs["camera_K"],
            ),
        )
        rot, trans = decompose_homogeneous_matrix(obs["camera_pose"])
        rr.log("world/head_camera", rr.Transform3D(translation=trans, mat3x3=rot, axis_length=0.3))

    def log_robot_xyt(self, obs):
        """Log robot world pose"""
        rr.set_time_seconds('realtime', time.time())
        xy = obs["gps"]
        theta = obs["compass"]
        rb_arrow = rr.Arrows3D(
            origins=[0, 0, 0],
            vectors=[0.5, 0, 0],
            radii=0.02,
            labels="robot",
            colors=[255, 0, 0, 255],
        )
        rr.log("world/robot/arrow", rb_arrow)
        rr.log("world/robot/blob", rr.Points3D([0, 0, 0], colors=[255, 0, 0, 255], radii=0.175))
        rr.log(
            "world/robot",
            rr.Transform3D(
                translation=[xy[0], xy[1], 0],
                rotation=rr.RotationAxisAngle(axis=[0, 0, 1], radians=theta),
                axis_length=0.7,
            ),
        )

    def log_ee_frame(self, obs):
        """log end effector pose"""
        rr.set_time_seconds('realtime', time.time())
        # EE Frame
        rot, trans = decompose_homogeneous_matrix(obs["ee_pose"])
        ee_arrow = rr.Arrows3D(
            origins=[0, 0, 0], vectors=[0.2, 0, 0], radii=0.02, labels="ee", colors=[0, 255, 0, 255]
        )
        rr.log("world/ee/arrow", ee_arrow)
        rr.log("world/ee", rr.Transform3D(translation=trans, mat3x3=rot, axis_length=0.3))

    def log_ee_camera(self, servo):
        """Log end effector camera pose and images"""
        rr.set_time_seconds('realtime', time.time())
        # EE Camera
        rr.log("world/ee_camera/rgb", rr.Image(servo.ee_rgb))
        rr.log("world/ee_camera/depth", rr.DepthImage(servo.ee_depth, meter=1000.0))
        rr.log(
            "world/ee_camera/rgb",
            rr.Pinhole(
                resolution=[servo.ee_rgb.shape[1], servo.ee_rgb.shape[0]],
                image_from_camera=servo.ee_camera_K,
            ),
        )
        rot, trans = decompose_homogeneous_matrix(servo.ee_camera_pose)
        rr.log("world/ee_camera", rr.Transform3D(translation=trans, mat3x3=rot, axis_length=0.3))

    def update_voxel_map(self, space):
        """Log voxel map (SparseVoxelMapNavigationSpace)"""
        rr.set_time_seconds('realtime', time.time())
        points, _, _, rgb = space.voxel_map.voxel_pcd.get_pointcloud()
        rr.log(
            "world/point_cloud",
            rr.Points3D(positions=points, radii=np.ones(rgb.shape[0]) * 0.01, colors=np.int64(rgb)),
        )
        grid_origin = space.voxel_map.grid_origin
        obstacles, explored = space.voxel_map.get_2d_map()
        grid_resolution = space.voxel_map.grid_resolution
        obs_points = np.array(occupancy_map_to_3d_points(obstacles, grid_origin, grid_resolution))
        explored_points = np.array(
            occupancy_map_to_3d_points(explored, grid_origin, grid_resolution)
        )
        rr.log(
            "world/obstacles",
            rr.Points3D(
                positions=obs_points, radii=np.ones(points.shape[0]) * 0.025, colors=[255, 0, 0]
            ),
        )
        rr.log(
            "world/explored",
            rr.Points3D(
                positions=explored_points,
                radii=np.ones(points.shape[0]) * 0.01,
                colors=[255, 255, 255],
            ),
        )

    def update_scene_graph(self, scene_graph, semantic_sensor=None):
        """Log objects bounding boxes and relationships
        Args:
            scene_graph (SceneGraph): Scene graph object
            semantic_sensor (OvmmPerception): Semantic sensor object
        """
        if semantic_sensor:
            rr.set_time_seconds('realtime', time.time())
            centers = []
            labels = []
            bounds = []
            colors = []

            for idx, instance in enumerate(scene_graph.instances):
                name = semantic_sensor.get_class_name_for_id(instance.category_id)
                if name not in self.bbox_colors_memory:
                    self.bbox_colors_memory[name] = np.random.randint(0, 255, 3)
                best_view = instance.get_best_view()
                bbox_bounds = best_view.bounds  # 3D Bounds
                half_sizes = [(b[0] - b[1]) / 2 for b in bbox_bounds]
                bounds.append(half_sizes)
                pose = scene_graph.get_ins_center_pos(idx)
                confidence = best_view.score
                centers.append(pose)
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
            )

    def step(self, obs, servo):
        """Log all the data"""
        if obs and servo:
            rr.set_time_seconds('realtime', time.time())
            try:
                self.log_robot_xyt(obs)
                self.log_head_camera(obs)
                self.log_ee_frame(obs)
                self.log_ee_camera(servo)
            except Exception as e:
                print(e)
