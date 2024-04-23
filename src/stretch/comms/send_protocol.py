import zmq

from stretch.versions import __stretchpy_protocol__ as STRETCHPY_PROTOCOL


def initialize(port):
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://*:{port}")
    poll = zmq.Poller()
    poll.register(sock, flags=zmq.POLLIN)
    return sock, poll


def send_spp(sock, poll):
    socks = dict(poll.poll(20.0))  # 50hz
    if sock in socks and socks[sock] == zmq.POLLIN:
        # message = sock.recv_string()
        sock.send_string(STRETCHPY_PROTOCOL)
