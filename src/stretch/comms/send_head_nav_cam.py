import io

import cv2
import zmq

import stretch.drivers.head_nav_cam as driver


def initialize(camarr_port, camb64_port):
    # zeromq
    ctx = zmq.Context()
    camarr_sock = ctx.socket(zmq.PUB)
    camarr_sock.setsockopt(zmq.SNDHWM, 1)
    camarr_sock.setsockopt(zmq.RCVHWM, 1)
    camarr_sock.bind(f"tcp://*:{camarr_port}")
    camb64_sock = ctx.socket(zmq.PUB)
    camb64_sock.setsockopt(zmq.SNDHWM, 1)
    camb64_sock.setsockopt(zmq.RCVHWM, 1)
    camb64_sock.bind(f"tcp://*:{camb64_port}")

    # opencv camera
    camera = driver.HeadNavCam()

    return camarr_sock, camb64_sock, camera


def send_imagery_as_numpy_arr(sock, camera):
    sock.send_pyobj(camera.get_image())


def send_imagery_as_base64_str(sock, camera):
    did_encode, buffer = cv2.imencode(".jpg", camera.get_image())
    if not did_encode:
        print("Warning: failed to encode head nav cam image. Skipping...")
        return
    io_buffer = io.BytesIO(buffer)
    sock.send(io_buffer.read())
