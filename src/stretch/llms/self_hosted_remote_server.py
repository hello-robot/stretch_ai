# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import zmq

from stretch.llms.base import AbstractLLMClient


class SelfHostedRemoteServer:
    """Class which hosts an LLM client on your own machine, locally or remotely, in order to give you the GPU to run your own LLM."""

    def __init__(self, client: AbstractLLMClient):
        self._client = client
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        self._socket.bind("tcp://*:5555")

    def run(self):
        while True:
            #  Wait for next request from client
            message = self._socket.recv()
            print(f"Received request: {message}")

            #  Do some 'work'
            try:
                plan = self._client(message.decode())
                response = {"plan": plan}
            except Exception as e:
                response = {"error": str(e)}

            #  Send reply back to client
            self._socket.send_json(response)

    def __del__(self):
        self._socket.close()
        self._context.term()
