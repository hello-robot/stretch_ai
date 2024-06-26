import zmq


class CommsNode:
    """Stretch comms"""

    def __init__(self):
        self.context = zmq.Context()

    def _make_pub_socket(self, send_port, use_remote_computer: bool = True):
        socket = self.context.socket(zmq.PUB)
        socket.setsockopt(zmq.SNDHWM, 1)
        socket.setsockopt(zmq.RCVHWM, 1)

        if use_remote_computer:
            send_address = "tcp://*:" + str(send_port)
        else:
            desktop_ip = "127.0.0.1"
            send_address = f"tcp://{desktop_ip}:" + str(send_port)

        print(f"Publishing on {send_address}...")
        socket.bind(send_address)
        return socket

    def _make_sub_socket(self, recv_port, use_remote_computer: bool = True):

        # Set up the receiver/subscriber using ZMQ
        recv_socket = self.context.socket(zmq.SUB)
        recv_socket.setsockopt(zmq.SUBSCRIBE, b"")
        recv_socket.setsockopt(zmq.SNDHWM, 1)
        recv_socket.setsockopt(zmq.RCVHWM, 1)
        recv_socket.setsockopt(zmq.CONFLATE, 1)

        # Make connections
        if use_remote_computer:
            recv_address = "tcp://*:" + str(recv_port)
        else:
            desktop_ip = "127.0.0.1"
            recv_address = f"tcp://{desktop_ip}:" + str(recv_port)

        print(f"Listening on {recv_address}...")
        recv_socket.bind(recv_address)
        return recv_socket, recv_address
