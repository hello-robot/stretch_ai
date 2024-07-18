#!/usr/bin/env python3

import numpy as np
import rerun as rr  # pip install rerun-sdk
import cv2


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
            rr.Boxes3D(half_sizes=[4, 4, 1], centers=[0, 0, 1], colors=[255, 255, 255, 255]),
            static=True,
        )
    def step(self, obs):
        if obs:
            try:
                xy = obs['gps']
                theta = obs['compass']
                print(f"Logging XYT: {xy},{theta}")
                arrow = rr.Arrows3D(origins=[0, 0, 0], vectors=[1, 0, 0], radii=0.02, labels="robot", colors=[255, 0, 0, 255])
                rr.log("world/robot/arrow", arrow)
                rr.log("world/robot/blob", rr.Points3D([0,0,0], colors=[255, 0, 0, 255], radii=0.175))
                rr.log("world/robot", rr.Transform3D(translation=[xy[0], xy[1], 0], rotation=rr.RotationAxisAngle(axis=[0,0,1],radians=theta)))
            except Exception as e:
                pass
        
