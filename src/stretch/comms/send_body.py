import zmq
import json
import stretch_body.robot


def initialize(port):
    # zeromq
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.setsockopt(zmq.SNDHWM, 1)
    sock.setsockopt(zmq.RCVHWM, 1)
    sock.bind(f"tcp://*:{port}")

    # stretch body
    robot = stretch_body.robot.Robot()
    robot.startup()
    is_runstopped = robot.status['pimu']['runstop_event']
    if is_runstopped:
        robot.pimu.runstop_event_reset()
        robot.push_command()
    is_homed = robot.is_homed()
    if not is_homed:
        robot.home()

    return sock, robot


def send_status(sock, robot):
    sock.send_json(json.dumps(robot.get_status()))


def send_parameters(sock, robot):
    pass


def send_urdf(sock, robot):
    pass
