# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import cv2


class HeadNavCam:
    def __init__(
        self,
        port="/dev/hello-nav-head-camera",
        imgformat="MJPG",
        size=[800, 600],
        fps=100,
        brightness=10,
        contrast=30,
        saturation=80,
        hue=0,
        gamma=80,
        gain=10,
        white_balance_temp=4600,
        sharpness=3,
        backlight=1,
    ):
        self.port = port
        self.imgformat = imgformat
        self.size = size
        self.fps = fps
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.gamma = gamma
        self.gain = gain
        self.white_balance_temp = white_balance_temp
        self.sharpness = sharpness
        self.backlight = backlight

        self.cap = cv2.VideoCapture(self.port)
        self.update_camera_settings()

    def update_camera_settings(self):
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*self.imgformat))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, self.brightness)
        self.cap.set(cv2.CAP_PROP_CONTRAST, self.contrast)
        self.cap.set(cv2.CAP_PROP_SATURATION, self.saturation)
        self.cap.set(cv2.CAP_PROP_HUE, self.hue)
        self.cap.set(cv2.CAP_PROP_GAMMA, self.gamma)
        self.cap.set(cv2.CAP_PROP_GAIN, self.gain)
        self.cap.set(cv2.CAP_PROP_WB_TEMPERATURE, self.white_balance_temp)
        self.cap.set(cv2.CAP_PROP_SHARPNESS, self.sharpness)
        self.cap.set(cv2.CAP_PROP_BACKLIGHT, self.backlight)

    def get_image(self):
        _, img = self.cap.read()
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return img

    def stream(self):
        while True:
            yield self.get_image()
