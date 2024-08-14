# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import time
from pathlib import Path

import cv2
import webcam as wc

camera_name = "Logitech Webcam C930e"
image_width = 1920
image_height = 1080
fps = 30

num_images_to_collect = 60
time_between_images_sec = 0.5
image_directory = wc.get_calibration_directory(camera_name, image_width, image_height)
image_base_name = "webcam_calibration_image"

Path(image_directory).mkdir(parents=True, exist_ok=True)

webcam = wc.Webcam(
    camera_name=camera_name,
    fps=fps,
    image_width=image_width,
    image_height=image_height,
    use_calibration=False,
    show_images=False,
)

prev_save_time = time.time()
num_images = 0

while num_images < num_images_to_collect:

    color_image, camera_info = webcam.get_next_frame()

    cv2.imshow("image from camera", color_image)
    cv2.waitKey(1)

    curr_time = time.time()

    if (curr_time - prev_save_time) > time_between_images_sec:
        num_images = num_images + 1
        file_name = image_directory + image_base_name + "_" + str(num_images).zfill(4) + ".png"
        print("save", file_name)
        cv2.imwrite(file_name, color_image)
        prev_save_time = curr_time
