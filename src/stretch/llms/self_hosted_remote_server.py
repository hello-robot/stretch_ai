# Copyright 2024 Hello Robot Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# licence information maybe found below, if so.

import zmq

from stretch.llms.base import AbstractLLMClient, AbstractPrompt


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
