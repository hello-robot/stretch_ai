#!/usr/bin/env python

import argparse
import pprint as pp
import threading
import time

import cv2
import numpy as np
import stretch_body.robot as rb
import zmq

import stretch.app.dex_teleop.dex_teleop_parameters as dt
from stretch.app.dex_telep.follower import DexTeleopFollower


def main(args):
    use_fastest_mode = args.fast
    manipulate_on_ground = args.ground
    left_handed = args.left
    using_stretch_2 = args.stretch_2
    slide_lift_range = args.slide_lift_range

    # The 'default', 'slow', 'fast', and 'max' options are defined by
    # Hello Robot. The 'fastest_stretch_2' option has been specially tuned for
    # this application.
    #
    # WARNING: 'fastest_stretch_*' have velocities and accelerations that exceed
    # the factory 'max' values defined by Hello Robot.
    if use_fastest_mode:
        if using_stretch_2:
            robot_speed = "fastest_stretch_2"
        else:
            robot_speed = "fastest_stretch_3"
    else:
        robot_speed = "slow"

    # Note on control here
    print("Running with robot_speed =", robot_speed)

    follower = DexTeleopFollower(
        robot_speed,
        manipulate_on_ground=manipulate_on_ground,
        robot_allowed_to_move=True,
        using_stretch_2=using_stretch_2,
        scaling=args.scaling,
        gamma=args.gamma,
        exposure=args.exposure,
        send_port=args.send_port,
        recv_port=args.recv_port,
        look_at_ee=True,
    )
    follower.spin()


if __name__ == "__main__":

    args = dt.get_arg_parser().parse_args()
    main(args)
