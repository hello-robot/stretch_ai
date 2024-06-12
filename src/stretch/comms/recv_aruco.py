import zmq

from stretch.exceptions.aruco.database import AddToArucoDatabaseError, DeleteFromArucoDatabaseError


def initialize(ip_addr, info_port, add_port, delete_port):
    ctx = zmq.Context()
    info_sock = ctx.socket(zmq.REQ)
    info_sock.connect(f"tcp://{ip_addr}:{info_port}")
    add_sock = ctx.socket(zmq.REQ)
    add_sock.connect(f"tcp://{ip_addr}:{add_port}")
    delete_sock = ctx.socket(zmq.REQ)
    delete_sock.connect(f"tcp://{ip_addr}:{delete_port}")
    return info_sock, add_sock, delete_sock


def recv_marker_info(sock):
    sock.send_string("requesting_marker_info")
    return sock.recv_pyobj()


def send_add_request(sock, marker):
    sock.send_pyobj(marker)
    add_status = sock.recv_string()
    if add_status != "Accepted" and add_status.split(": ", 1)[1:]:
        raise AddToArucoDatabaseError(add_status.split(": ", 1)[1])


def send_delete_request(sock, marker_id):
    sock.send_string(marker_id)
    delete_status = sock.recv_string()
    if delete_status != "Accepted" and delete_status.split(": ", 1)[1:]:
        raise DeleteFromArucoDatabaseError(delete_status.split(": ", 1)[1])
