#!/usr/bin/env python3

import numpy as np
import rerun as rr  # pip install rerun-sdk
import cv2


def decompose_homogeneous_matrix(homogeneous_matrix):
    if homogeneous_matrix.shape != (4, 4):
        raise ValueError("Input matrix must be 4x4")
    rotation_matrix = homogeneous_matrix[:3, :3]
    translation_vector = homogeneous_matrix[:3, 3]
    return rotation_matrix, translation_vector

class RerunVsualizer:
    def __init__(self) -> None:

        # Define a mapping for transforms
        self.path_to_frame = {
            "world": "map",
            "world/robot": "base_footprint",
            "world/robot/scan": "laser",
            "world/robot/head_camera": "camera_color_optical_frame",
            "world/robot/head_camera/points": "camera_depth_frame",
            "world/robot/ee_camera": "gripper_camera_color_optical_frame",
            "world/robot/ee_camera/points": "gripper_camera_depth_frame",
        }
        rr.init("Stretch_robot",spawn=False)
        rr.serve(open_browser=False,server_memory_limit="1GB")
        # Log a bounding box as a visual placeholder for the map
        rr.log(
            "map/box",
            rr.Boxes3D(half_sizes=[10, 10, 3], centers=[0, 0, 2], colors=[255, 255, 255, 255]),
            static=True,
        )
    def step(self, obs):
        if obs:
            try:
                # Robot World pose
                xy = obs['gps']
                theta = obs['compass']
                arrow = rr.Arrows3D(origins=[0, 0, 0], vectors=[1, 0, 0], radii=0.02, labels="robot", colors=[255, 0, 0, 255])
                rr.log("world/robot/arrow", arrow)
                rr.log("world/robot/blob", rr.Points3D([0,0,0], colors=[255, 0, 0, 255], radii=0.175))
                rr.log("world/robot", rr.Transform3D(translation=[xy[0], xy[1], 0], rotation=rr.RotationAxisAngle(axis=[0,0,1],radians=theta)))

                # Head Image
                rr.log("world/robot/head_camera/img", rr.Image(obs["rgb"]))
                rr.log("world/robot/head_camera/depth", rr.DepthImage(obs["depth"], meter=1000.0, colormap="viridis"))
                rr.log(
                    "world/robot/head_camera/img",
                    rr.Pinhole(
                        resolution=[obs["rgb"].shape[1], obs["rgb"].shape[0]],
                        image_from_camera=obs['camera_K'],
                    ),
                )
                rot,trans= decompose_homogeneous_matrix(obs['camera_pose'])
                rr.log("world/robot/head_camera", rr.Transform3D(translation=trans, mat3x3=rot))
            except Exception as e:
                print(e)
        
