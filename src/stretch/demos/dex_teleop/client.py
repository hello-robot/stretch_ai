import argparse
import sys
import time
from copy import deepcopy
from typing import Optional

import cv2
import numpy as np
import zmq

import stretch.utils.loop_stats as lt
from stretch.core import Evaluator


class RobotClient:
    """Simple remote desktop client."""

    def __init__(
        self,
        use_remote_computer: bool = True,
        robot_ip: Optional[str] = None,
        d405_port: int = 4405,
        verbose: bool = False,
        name: str = "robot_client",
    ):
        self.verbose = verbose
        if verbose:
            print("cv2.__version__ =", cv2.__version__)
            print("cv2.__path__ =", cv2.__path__)
            print("sys.version =", sys.version)
            print(f"{name=}  RobotClient.__init__")

        self.name = name

        d405_context = zmq.Context()
        d405_socket = d405_context.socket(zmq.SUB)
        d405_socket.setsockopt(zmq.SUBSCRIBE, b"")
        d405_socket.setsockopt(zmq.SNDHWM, 1)
        d405_socket.setsockopt(zmq.RCVHWM, 1)
        d405_socket.setsockopt(zmq.CONFLATE, 1)
        if use_remote_computer:
            d405_address = "tcp://" + robot_ip + ":" + str(d405_port)
        else:
            d405_address = "tcp://" + "127.0.0.1" + ":" + str(d405_port)

        d405_socket.connect(d405_address)

        self.d405_socket = d405_socket
        self.d405_address = d405_address
        self.d405_context = d405_context

    def run(self, evaluator: Evaluator = None):
        """Run the client. Optionally pass in an eval function."""

        loop_timer = lt.LoopStats(self.name)
        first_frame = True

        try:
            while True:

                loop_timer.mark_start()

                d405_output = self.d405_socket.recv_pyobj()
                color_image = d405_output["color_image"]
                depth_image = d405_output["depth_image"]
                depth_camera_info = d405_output["depth_camera_info"]
                depth_scale = d405_output["depth_scale"]

                if first_frame and evaluator is not None:
                    evaluator.set_camera_parameters(depth_camera_info, depth_scale)
                    first_frame = False

                # After extracting image, pass it to whatever is going to use it.
                # Results are currently ignored.
                # TODO: we might want to change this code path to something a bit better.
                _ = evaluator.apply(color_image, depth_image)

                loop_timer.mark_end()
                if self.verbose:
                    loop_timer.pretty_print()

                if evaluator.is_done():
                    break

        finally:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot_ip", type=str, default="192.168.1.15")
    parser.add_argument("--d405_port", type=int, default=4405)
    args = parser.parse_args()

    client = RobotClient(
        use_remote_computer=True,
        robot_ip=args.robot_ip,
        d405_port=args.d405_port,
        verbose=True,
    )
    # Default evaluator will just show images
    evaluator = Evaluator()
    client.run(evaluator)
