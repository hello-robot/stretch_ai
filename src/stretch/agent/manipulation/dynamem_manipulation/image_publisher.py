# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import cv2
import numpy as np


def send_array(socket, A, flags=0, copy=True, track=False):
    """send a numpy array with metadata"""
    return socket.send_pyobj(A)


def recv_array(socket, flags=0, copy=True, track=False):
    """recv a numpy array"""
    return socket.recv_pyobj()


def send_rgb_img(socket, img):
    img = img.astype(np.uint8)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    _, img_encoded = cv2.imencode(".jpg", img, encode_param)
    socket.send(img_encoded.tobytes())


def send_depth_img(socket, depth_img):
    depth_img = (depth_img * 1000).astype(np.uint16)
    encode_param = [
        int(cv2.IMWRITE_PNG_COMPRESSION),
        3,
    ]  # Compression level from 0 (no compression) to 9 (max compression)
    _, depth_img_encoded = cv2.imencode(".png", depth_img, encode_param)
    socket.send(depth_img_encoded.tobytes())


class DynamemCamera:
    def __init__(self, robot):
        self.robot = robot

        # Camera intrinsics
        intrinsics = self.robot.get_camera_K()
        self.fy = intrinsics[0, 0]
        self.fx = intrinsics[1, 1]
        self.cy = intrinsics[0, 2]
        self.cx = intrinsics[1, 2]
        print(self.fx, self.fy, self.cx, self.cy)

        # selected ix and iy coordinates
        self.ix, self.iy = None, None

    def capture_image(self):
        self.rgb_image, self.depth_image, self.points = self.robot.get_images(compute_xyz=True)
        self.rgb_image = np.rot90(self.rgb_image, k=1)[:, :, [2, 1, 0]]
        self.depth_image = np.rot90(self.depth_image, k=1)
        self.points = np.rot90(self.points, k=1)

        self.rgb_image = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2RGB)

        return self.rgb_image, self.depth_image, self.points


class ImagePublisher:
    def __init__(self, robot, socket):
        self.camera = DynamemCamera(robot)
        self.socket = socket

    def publish_image(self, text, mode, head_tilt=-1):
        image, depth, points = self.camera.capture_image()
        # camera_pose = self.camera.robot.head.get_pose_in_base_coords()

        rotated_image = np.rot90(image, k=-1)
        rotated_depth = np.rot90(depth, k=-1)
        rotated_point = np.rot90(points, k=-1)

        ## Send RGB, depth and camera intrinsics data
        send_rgb_img(self.socket, rotated_image)
        print(self.socket.recv_string())
        send_depth_img(self.socket, rotated_depth)
        print(self.socket.recv_string())
        send_array(
            self.socket,
            np.array(
                [
                    self.camera.fy,
                    self.camera.fx,
                    self.camera.cy,
                    self.camera.cx,
                    int(head_tilt * 100),
                ]
            ),
        )
        print(self.socket.recv_string())

        ## Sending Object text and Manipulation mode
        self.socket.send_string(text)
        print(self.socket.recv_string())
        self.socket.send_string(mode)
        print(self.socket.recv_string())

        ## Waiting for the base and camera transforms to center the object vertically and horizontally
        self.socket.send_string("Waiting for gripper pose/ base and head trans from workstation")
        translation = recv_array(self.socket)
        self.socket.send_string("translation received by robot")
        rotation = recv_array(self.socket)
        self.socket.send_string("rotation received by robot")
        add_data = recv_array(self.socket)
        self.socket.send_string(f"Additional data received robot")

        depth = add_data[0]
        width = add_data[1]
        retry = add_data[2]
        print(f"Additional data received - {add_data}")
        print("translation: ")
        print(translation)
        print("rotation: ")
        print(rotation)
        print(self.socket.recv_string())
        return translation, rotation, depth, width, retry
