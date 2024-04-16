import zmq
import json


def initialize(ip_addr, port):
    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.SUBSCRIBE, b'')
    sock.setsockopt(zmq.SNDHWM, 1)
    sock.setsockopt(zmq.RCVHWM, 1)
    sock.setsockopt(zmq.CONFLATE, 1)
    sock.connect(f"tcp://{ip_addr}:{port}")
    return sock


def recv_status(sock):
    return json.loads(sock.recv_json())
