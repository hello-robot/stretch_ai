#!/usr/bin/env python3

import numpy as np
import rerun as rr  # pip install rerun-sdk


def decompose_homogeneous_matrix(homogeneous_matrix):
    if homogeneous_matrix.shape != (4, 4):
        raise ValueError("Input matrix must be 4x4")
    rotation_matrix = homogeneous_matrix[:3, :3]
    translation_vector = homogeneous_matrix[:3, 3]
    return rotation_matrix, translation_vector


class RerunVsualizer:
    def __init__(self):
        rr.init("Stretch_robot", spawn=False)
        rr.serve(open_browser=False, server_memory_limit="1GB")
        # Create environment Box place holder
        rr.log(
            "map/box",
            rr.Boxes3D(half_sizes=[10, 10, 3], centers=[0, 0, 2], colors=[255, 255, 255, 255]),
            static=True,
        )

    def log_head_camera(self, obs):
        # Head Camera
        rr.log("world/head_camera/img", rr.Image(obs["rgb"]))
        rr.log("world/head_camera/depth", rr.DepthImage(obs["depth"], meter=1000.0))
        rr.log(
            "world/head_camera/img",
            rr.Pinhole(
                resolution=[obs["rgb"].shape[1], obs["rgb"].shape[0]],
                image_from_camera=obs["camera_K"],
            ),
        )
        rot, trans = decompose_homogeneous_matrix(obs["camera_pose"])
        rr.log("world/head_camera", rr.Transform3D(translation=trans, mat3x3=rot))

    def log_robot_xyt(self, obs):
        # Robot World pose
        xy = obs["gps"]
        theta = obs["compass"]
        rb_arrow = rr.Arrows3D(
            origins=[0, 0, 0],
            vectors=[1, 0, 0],
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
            ),
        )

    def log_ee_frame(self, servo):
        # EE Frame
        rot, trans = decompose_homogeneous_matrix(servo.ee_camera_pose)
        ee_arrow = rr.Arrows3D(
            origins=[0, 0, 0], vectors=[1, 0, 0], radii=0.02, labels="ee", colors=[0, 255, 0, 255]
        )
        rr.log("world/ee/arrow", ee_arrow)
        rr.log("world/ee/blob", rr.Points3D([0, 0, 0], colors=[0, 255, 0, 255], radii=0.09))
        rr.log("world/ee", rr.Transform3D(translation=trans, mat3x3=rot))

    def log_ee_camera(self, servo):
        # EE Camera
        rr.log("world/ee_camera/img", rr.Image(servo.ee_rgb))
        rr.log("world/ee_camera/depth", rr.DepthImage(servo.ee_depth, meter=1000.0))
        rr.log(
            "world/ee_camera/img",
            rr.Pinhole(
                resolution=[servo.ee_rgb.shape[1], servo.ee_rgb.shape[0]],
                image_from_camera=servo.ee_camera_K,
            ),
        )
        rot, trans = decompose_homogeneous_matrix(servo.ee_camera_pose)
        rr.log("world/ee_camera", rr.Transform3D(translation=trans, mat3x3=rot, axis_length=0.3))

    def step(self, obs, servo):
        if obs:
            try:
                self.log_robot_xyt(obs)
                self.log_head_camera(obs)
                self.log_ee_frame(servo)
                self.log_ee_camera(servo)
            except Exception as e:
                print(e)
