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

exposure_keywords = ["low", "medium", "auto"]
exposure_range = [0, 500000]


class D405(Realsense):
    def __init__(self, exposure):
        self.pipeline, self.profile = start_d405(exposure)

        print("Connecting to D405 and getting camera info...")
        self.depth_camera_info, self.color_camera_info = self.read_camera_infos()
        print(f"  depth camera: {self.depth_camera_info}")
        print(f"  color camera: {self.color_camera_info}")

    def get_camera_infos(self):
        return self.depth_camera_info, self.color_camera_info

    def get_images(self) -> Tuple[np.ndarray, np.ndarray]:
        depth_frame, color_frame = self.get_frames()
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        return depth_image, color_image

    def get_message(self) -> dict:
        """Get a message that can be sent via ZMQ"""
        depth_camera_info, color_camera_info = self.get_camera_infos()
        depth_scale = self.get_depth_scale()
        depth_image, color_image = self.get_images()
        d405_output = {
            "depth_camera_info": depth_camera_info,
            "color_camera_info": color_camera_info,
            "depth_scale": depth_scale,
            "depth_image": depth_image,
            "color_image": color_image,
        }
        return d405_output

    def stop(self):
        """Close everything down so we can end cleanly."""
        self.pipeline.stop()


def exposure_argument_is_valid(value):
    if value in exposure_keywords:
        return True
    is_string = isinstance(value, str)
    is_int = isinstance(value, int)
    int_value = exposure_range[0] - 10
    if is_string:
        if not value.isdigit():
            return False
        else:
            int_value = int(value)
    elif is_int:
        int_value = value
    if (int_value >= exposure_range[0]) or (int_value <= exposure_range[1]):
        return True
    return False


def check_exposure_value(value):
    if not exposure_argument_is_valid(value):
        raise ValueError(
            f"The provided exposure setting, {value}, is not a valid keyword, {exposure_keywords}, or is outside of the allowed numeric range, {exposure_range}."
        )


def prepare_exposure_value(value):
    check_exposure_value(value)
    if value in exposure_keywords:
        return value
    is_int = isinstance(value, int)
    if is_int:
        return value
    is_string = isinstance(value, str)
    if is_string:
        return int(value)
    return None


def start_d405(exposure):
    camera_info = [
        {
            "name": device.get_info(rs.camera_info.name),
            "serial_number": device.get_info(rs.camera_info.serial_number),
        }
        for device in rs.context().devices
    ]

    exposure = prepare_exposure_value(exposure)

    print("All cameras that were found:")
    print(camera_info)
    print()

    d405_info = None
    for info in camera_info:
        if info["name"].endswith("D405"):
            d405_info = info
    if d405_info is None:
        print("D405 camera not found")
        print("Exiting")
        exit()
    else:
        print("D405 found:")
        print(d405_info)
        print()

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(d405_info["serial_number"])

    # 1280 x 720, 5 fps
    # 848 x 480, 10 fps
    # 640 x 480, 30 fps

    # WIDTH, HEIGHT, FPS = 1280, 720, 5
    # WIDTH, HEIGHT, FPS = 848, 480, 10
    # WIDTH, HEIGHT, FPS = 640, 480, 30
    WIDTH, HEIGHT, FPS = 640, 480, 15
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)

    profile = pipeline.start(config)

    if exposure == "auto":
        # Use autoexposre
        stereo_sensor = pipeline.get_active_profile().get_device().query_sensors()[0]
        stereo_sensor.set_option(rs.option.enable_auto_exposure, True)
    else:
        default_exposure = 33000
        if exposure == "low":
            exposure_value = int(default_exposure / 3.0)
        elif exposure == "medium":
            exposure_value = 30000
        else:
            exposure_value = int(exposure)

        stereo_sensor = pipeline.get_active_profile().get_device().query_sensors()[0]
        stereo_sensor.set_option(rs.option.exposure, exposure_value)

    return pipeline, profile
