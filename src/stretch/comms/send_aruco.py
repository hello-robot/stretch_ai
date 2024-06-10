import zmq

import stretch.drivers.aruco_markers as driver


def initialize(info_port, add_port, delete_port):
    # zeromq
    ctx = zmq.Context()
    info_sock = ctx.socket(zmq.REP)
    info_sock.bind(f"tcp://*:{info_port}")
    info_poll = zmq.Poller()
    info_poll.register(info_sock, flags=zmq.POLLIN)
    add_sock = ctx.socket(zmq.REP)
    add_sock.bind(f"tcp://*:{add_port}")
    add_poll = zmq.Poller()
    add_poll.register(add_sock, flags=zmq.POLLIN)
    delete_sock = ctx.socket(zmq.REP)
    delete_sock.bind(f"tcp://*:{delete_port}")
    delete_poll = zmq.Poller()
    delete_poll.register(delete_sock, flags=zmq.POLLIN)

    # load marker database
    database = driver.MarkersDatabase()

    return info_sock, info_poll, add_sock, add_poll, delete_sock, delete_poll, database


def send_marker_info(sock, poll, db):
    socks = dict(poll.poll(40.0)) # 25hz
    if not (sock in socks and socks[sock] == zmq.POLLIN):
        return

    _ = sock.recv_string()
    sock.send_pyobj(db.marker_info)


def add_marker(sock, poll, db):
    socks = dict(poll.poll(40.0)) # 25hz
    if not (sock in socks and socks[sock] == zmq.POLLIN):
        return

    result = db.add(sock.recv_pyobj())
    sock.send_string(result)


def delete_marker(sock, poll, db):
    socks = dict(poll.poll(40.0)) # 25hz
    if not (sock in socks and socks[sock] == zmq.POLLIN):
        return

    result = db.delete(sock.recv_string())
    sock.send_string(result)
