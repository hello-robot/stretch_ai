# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

from typing import Tuple

import zmq
from zmq import Socket


class BaseCommsNode:
    def __init__(self):
        self.context = zmq.Context()

    def _new_pub_socket(self) -> Socket:
        socket = self.context.socket(zmq.PUB)
        socket.setsockopt(zmq.SNDHWM, 1)
        socket.setsockopt(zmq.RCVHWM, 1)
        return socket

    def _new_sub_socket(self) -> Socket:
        recv_socket = self.context.socket(zmq.SUB)
        recv_socket.setsockopt(zmq.SUBSCRIBE, b"")
        recv_socket.setsockopt(zmq.SNDHWM, 1)
        recv_socket.setsockopt(zmq.RCVHWM, 1)
        recv_socket.setsockopt(zmq.CONFLATE, 1)
        return recv_socket


class CommsNode(BaseCommsNode):
    """Stretch comms"""

    def _make_pub_socket(self, send_port: int, use_remote_computer: bool = True) -> Socket:
        socket = self._new_pub_socket()
        send_address = self._get_ip_address(send_port, use_remote_computer)

        print(f"Publishing on {send_address}...")
        socket.bind(send_address)
        return socket

    def _get_ip_address(self, port: int, use_remote_computer: bool = True) -> str:
        """Helper function to get IP addresses"""
        if use_remote_computer:
            recv_address = "tcp://*:" + str(port)
        else:
            desktop_ip = "127.0.0.1"
            recv_address = f"tcp://{desktop_ip}:" + str(port)
        return recv_address

    def _make_sub_socket(
        self, recv_port: int, use_remote_computer: bool = True
    ) -> Tuple[Socket, str]:

        # Set up the receiver/subscriber using ZMQ
        recv_socket = self._new_sub_socket()
        recv_address = self._get_ip_address(recv_port, use_remote_computer)

        print(f"Listening on {recv_address}...")
        recv_socket.bind(recv_address)
        return recv_socket, recv_address


class ClientCommsNode(BaseCommsNode):
    """Version of comms node which operates remotely."""

    def _get_ip_address(self, port: int, robot_ip: str, use_remote_computer: bool = True) -> str:
        # Use remote computer or whatever
        if use_remote_computer:
            address = "tcp://" + robot_ip + ":" + str(port)
        else:
            address = "tcp://" + "127.0.0.1" + ":" + str(port)
        return address

    def _make_pub_socket(
        self, send_port: int, robot_ip: str, use_remote_computer: bool = True
    ) -> Socket:
        socket = self._new_pub_socket()
        addr = self._get_ip_address(send_port, robot_ip, use_remote_computer)
        print(f"Publishing on {addr}...")
        socket.connect(addr)
        return socket

    def _make_sub_socket(
        self, recv_port: int, robot_ip: str, use_remote_computer: bool = True
    ) -> Tuple[Socket, str]:
        recv_socket = self._new_sub_socket()
        addr = self._get_ip_address(recv_port, robot_ip, use_remote_computer)
        print(f"Listening on {addr}...")
        recv_socket.connect(addr)
        return recv_socket, addr
