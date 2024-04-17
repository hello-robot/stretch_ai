import zmq


def initialize(ip_addr, camarr_port, camb64_port):
    ctx = zmq.Context()
    camarr_sock = ctx.socket(zmq.SUB)
    camarr_sock.setsockopt(zmq.SUBSCRIBE, b'')
    camarr_sock.setsockopt(zmq.SNDHWM, 1)
    camarr_sock.setsockopt(zmq.RCVHWM, 1)
    camarr_sock.setsockopt(zmq.CONFLATE, 1)
    camarr_sock.connect(f"tcp://{ip_addr}:{camarr_port}")
    camb64_sock = ctx.socket(zmq.SUB)
    camb64_sock.setsockopt(zmq.SUBSCRIBE, b'')
    camb64_sock.setsockopt(zmq.SNDHWM, 1)
    camb64_sock.setsockopt(zmq.RCVHWM, 1)
    camb64_sock.setsockopt(zmq.CONFLATE, 1)
    camb64_sock.connect(f"tcp://{ip_addr}:{camb64_port}")
    return camarr_sock, camb64_sock


def recv_imagery_as_numpy_arr(sock):
    return sock.recv_pyobj()
