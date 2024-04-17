import zmq
from stretch.exceptions.connection import MismatchedProtocolException
from stretch.version import __stretchpy_protocol__ as STRETCHPY_PROTOCOL


def initialize(ip_addr, port):
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.connect(f"tcp://{ip_addr}:{port}")
    return sock


def recv_spp(sock):
    sock.send_string("requesting_stretchpy_protocol")
    server_protocol = sock.recv_string()
    if server_protocol != STRETCHPY_PROTOCOL:
        raise MismatchedProtocolException()
