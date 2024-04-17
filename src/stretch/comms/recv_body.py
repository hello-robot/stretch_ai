import zmq
import json
from stretch.exceptions.motion import MoveByMotionNotAcceptedException


def initialize(ip_addr, status_port, moveby_port):
    ctx = zmq.Context()
    status_sock = ctx.socket(zmq.SUB)
    status_sock.setsockopt(zmq.SUBSCRIBE, b'')
    status_sock.setsockopt(zmq.SNDHWM, 1)
    status_sock.setsockopt(zmq.RCVHWM, 1)
    status_sock.setsockopt(zmq.CONFLATE, 1)
    status_sock.connect(f"tcp://{ip_addr}:{status_port}")
    moveby_sock = ctx.socket(zmq.REQ)
    moveby_sock.connect(f"tcp://{ip_addr}:{moveby_port}")
    return status_sock, moveby_sock


def recv_status(sock):
    return json.loads(sock.recv_json())


def send_moveby(sock, pose):
    sock.send_json(json.dumps(pose))
    moveby_status = sock.recv_string()
    if moveby_status != "Accepted" and moveby_status.split(': ', 1)[1:]:
        raise MoveByMotionNotAcceptedException(moveby_status.split(': ', 1)[1])
