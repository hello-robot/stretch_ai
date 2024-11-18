# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

from typing import Tuple

import numpy as np
import pyrealsense2 as rs

from stretch.drivers.realsense_base import Realsense

WIDTH, HEIGHT, FPS = 640, 480, 30
# WIDTH, HEIGHT, FPS = 640, 480, 15


class D435i(Realsense):
    """Wrapper for accessing data from a D435 realsense camera, used as the head camera on Stretch RE1, RE2, and RE3."""

    def __init__(self, exposure: str = "auto", camera_number: int = 0):
        print("Connecting to D435i and getting camera info...")
        self._setup_camera(exposure=exposure, number=camera_number)
        self.depth_camera_info, self.color_camera_info = self.read_camera_infos()
        print(f"  depth camera: {self.depth_camera_info}")
        print(f"  color camera: {self.color_camera_info}")

    def get_camera_infos(self):
        return self.depth_camera_info, self.color_camera_info

    def get_images(self, depth_type: type = np.float16) -> Tuple[np.ndarray, np.ndarray]:
        """Get a pair of numpy arrays for the images we want to use."""

        # Get the frames from the realsense
        frames = self.wait_for_frames()

        # Align the depth frame to color frame
        aligned_frames = self.align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        depth_image = np.asanyarray(aligned_depth_frame.get_data()).astype(depth_type)
        color_image = np.asanyarray(color_frame.get_data())
        return depth_image, color_image

    def _setup_camera(self, exposure: str = "auto", number: int = 0):
        """
        Args:
            number(int): which camera to pick in order.
        """

        camera_info = [
            {
                "name": device.get_info(rs.camera_info.name),
                "serial_number": device.get_info(rs.camera_info.serial_number),
            }
            for device in rs.context().devices
        ]
        print("Searching for D435i...")
        d435i_infos = []
        for i, info in enumerate(camera_info):
            print(i, info["name"], info["serial_number"])
            if "D435I" in info["name"]:
                d435i_infos.append(info)

        if len(d435i_infos) == 0:
            raise RuntimeError("could not find any supported d435i cameras")

        self.exposure = exposure
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Specifically enable the camera we want to use - make sure it's d435i
        self.config.enable_device(d435i_infos[number]["serial_number"])
        self.config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
        self.config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
        self.profile = self.pipeline.start(self.config)

        # Create an align object to align depth frames to color frames
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)

        if exposure == "auto":
            # Use autoexposre
            self.stereo_sensor = self.pipeline.get_active_profile().get_device().query_sensors()[0]
            self.stereo_sensor.set_option(rs.option.enable_auto_exposure, True)
        else:
            default_exposure = 33000
            if exposure == "low":
                exposure_value = int(default_exposure / 3.0)
            elif exposure == "medium":
                exposure_value = 30000
            else:
                exposure_value = int(exposure)

            self.stereo_sensor = self.pipeline.get_active_profile().get_device().query_sensors()[0]
            self.stereo_sensor.set_option(rs.option.exposure, exposure_value)


if __name__ == "__main__":
    camera = D435i()
    print(camera.get_message())
