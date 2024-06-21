import zmq

from stretch.exceptions.connection import MismatchedProtocolException, ServerNotFoundException
from stretch.versions import __stretchpy_protocol__ as STRETCHPY_PROTOCOL


def initialize(ip_addr, port):
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.connect(f"tcp://{ip_addr}:{port}")
    poll = zmq.Poller()
    poll.register(sock, flags=zmq.POLLIN)
    return sock, poll


def recv_spp(sock, poll):
    # send request
    sock.send_string("requesting_stretchpy_protocol")

    # wait for reply
    socks = dict(poll.poll(1000.0))
    if not (sock in socks and socks[sock] == zmq.POLLIN):
        raise ServerNotFoundException("Ensure Stretch is turned on")

    # process reply
    server_protocol = sock.recv_string()
    if server_protocol != STRETCHPY_PROTOCOL:
        raise MismatchedProtocolException()
