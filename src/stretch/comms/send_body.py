import json
import numbers

import stretch_body.robot
import zmq


def initialize(status_port, moveby_port):
    # zeromq
    ctx = zmq.Context()
    status_sock = ctx.socket(zmq.PUB)
    status_sock.setsockopt(zmq.SNDHWM, 1)
    status_sock.setsockopt(zmq.RCVHWM, 1)
    status_sock.bind(f"tcp://*:{status_port}")
    moveby_sock = ctx.socket(zmq.REP)
    moveby_sock.bind(f"tcp://*:{moveby_port}")
    moveby_poll = zmq.Poller()
    moveby_poll.register(moveby_sock, flags=zmq.POLLIN)

    # stretch body
    robot = stretch_body.robot.Robot()
    robot.startup()
    is_runstopped = robot.status["pimu"]["runstop_event"]
    if is_runstopped:
        robot.pimu.runstop_event_reset()
        robot.push_command()
    is_homed = robot.is_homed()
    if not is_homed:
        robot.home()

    return status_sock, moveby_sock, moveby_poll, robot


def send_status(sock, robot):
    sock.send_json(json.dumps(robot.get_status()))


def exec_moveby(sock, poll, robot):
    socks = dict(poll.poll(40.0))  # 25hz
    if not (sock in socks and socks[sock] == zmq.POLLIN):
        return

    pose = json.loads(sock.recv_json())

    if "joint_translate" in pose and "joint_rotate" in pose:
        sock.send_string(
            "Rejected: Cannot translate & rotate mobile base simultaneously"
        )
        return
    for joint, moveby_amount in pose.items():
        if not isinstance(moveby_amount, numbers.Real):
            sock.send_string(f"Rejected: Cannot move {joint} by {moveby_amount} amount")
            return

    if "joint_translate" in pose:
        robot.base.translate_by(pose["joint_translate"])
    if "joint_rotate" in pose:
        robot.base.rotate_by(pose["joint_rotate"])
    if "joint_lift" in pose:
        robot.lift.move_by(pose["joint_lift"])
    if "joint_arm" in pose:
        robot.arm.move_by(pose["joint_arm"])
    if "wrist_extension" in pose:
        robot.arm.move_by(pose["wrist_extension"])
    robot.push_command()
    if "joint_wrist_yaw" in pose:
        robot.end_of_arm.move_by("wrist_yaw", pose["joint_wrist_yaw"])
    if "joint_wrist_pitch" in pose:
        robot.end_of_arm.move_by("wrist_pitch", pose["joint_wrist_pitch"])
    if "joint_wrist_roll" in pose:
        robot.end_of_arm.move_by("wrist_roll", pose["joint_wrist_roll"])
    if "joint_gripper" in pose:
        robot.end_of_arm.move_by("stretch_gripper", pose["joint_gripper"])
    if "joint_head_pan" in pose:
        robot.head.move_by("head_pan", pose["joint_head_pan"])
    if "joint_head_tilt" in pose:
        robot.head.move_by("head_tilt", pose["joint_head_tilt"])
    sock.send_string("Accepted")


def send_parameters(sock, robot):
    pass


def send_urdf(sock, robot):
    pass
