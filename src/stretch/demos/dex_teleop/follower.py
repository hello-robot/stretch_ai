#!/usr/bin/env python

import argparse
import pprint as pp

import numpy as np
import zmq

import stretch.demos.dex_teleop.dex_teleop_parameters as dt
import stretch.demos.dex_teleop.gripper_to_goal as gg
import stretch.motion.simple_ik as si
import stretch.utils.loop_stats as lt

if __name__ == "__main__":

    args = dt.get_arg_parser().parse_args()
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
    print("running with robot_speed =", robot_speed)

    lift_middle = dt.get_lift_middle(manipulate_on_ground)
    center_configuration = dt.get_center_configuration(lift_middle)
    starting_configuration = dt.get_starting_configuration(lift_middle)

    # Initialize IK
    simple_ik = si.SimpleIK()

    # Define the center position for the wrist that corresponds with
    # the teleop origin.
    center_wrist_position = simple_ik.fk_rotary_base(center_configuration)

    gripper_to_goal = gg.GripperToGoal(
        robot_speed, starting_configuration, dt.robot_allowed_to_move, using_stretch_2
    )

    loop_timer = lt.LoopStats("dex_teleop_follower")
    print_timing = False
    print_goal = False

    goal_recv_context = zmq.Context()
    goal_recv_socket = goal_recv_context.socket(zmq.SUB)
    goal_recv_socket.setsockopt(zmq.SUBSCRIBE, b"")
    goal_recv_socket.setsockopt(zmq.SNDHWM, 1)
    goal_recv_socket.setsockopt(zmq.RCVHWM, 1)
    goal_recv_socket.setsockopt(zmq.CONFLATE, 1)
    # goal_recv_address = 'tcp://10.1.10.71:5555'
    goal_recv_address = "tcp://192.168.1.10:5555"
    goal_recv_socket.connect(goal_recv_address)

    while True:
        loop_timer.mark_start()
        goal_dict = goal_recv_socket.recv_pyobj()
        if goal_dict:
            if print_goal:
                print("goal_dict =")
                pp.pprint(goal_dict)
            gripper_to_goal.update_goal(**goal_dict)
        loop_timer.mark_end()
        if print_timing:
            loop_timer.pretty_print()
