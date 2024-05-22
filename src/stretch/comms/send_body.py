import json
import numbers

import zmq

import stretch.drivers.body as driver


def initialize(status_port, moveby_port, basevel_port):
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
    basevel_sock = ctx.socket(zmq.REP)
    basevel_sock.bind(f"tcp://*:{basevel_port}")
    basevel_poll = zmq.Poller()
    basevel_poll.register(basevel_sock, flags=zmq.POLLIN)

    # stretch body
    body = driver.Body()

    return status_sock, moveby_sock, moveby_poll, basevel_sock, basevel_poll, body


def send_status(sock, body):
    sock.send_json(json.dumps(body.get_status()))


def exec_moveby(sock, poll, body):
    socks = dict(poll.poll(40.0))  # 25hz
    if not (sock in socks and socks[sock] == zmq.POLLIN):
        return

    pose = sock.recv_json()
    if isinstance(pose, str):
        pose = json.loads(pose)

    if "joint_translate" in pose and "joint_rotate" in pose:
        sock.send_string("Rejected: Cannot translate & rotate mobile base simultaneously")
        return
    for joint, moveby_amount in pose.items():
        if not isinstance(moveby_amount, numbers.Real):
            sock.send_string(f"Rejected: Cannot move {joint} by {moveby_amount} amount")
            return

    result = body.move_by(pose)
    sock.send_string(result)


def exec_basevel(sock, poll, body):
    socks = dict(poll.poll(40.0))
    if not (sock in socks and socks[sock] == zmq.POLLIN):
        return

    twist = sock.recv_json()
    if isinstance(twist, str):
        twist = json.loads(twist)
    body.drive(twist)
    sock.send_string("Accepted")


def send_parameters(sock, body):
    pass


def send_urdf(sock, body):
    pass
