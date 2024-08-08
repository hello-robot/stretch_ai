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


class SelfHostedRemoteClient(AbstractLLMClient):
    """Client which runs on your own machine, locally or remotely, in order to give you the GPU to run your own LLM.
    It communicates with the LLM via a ZMQ socket. It will send and receive dictionaries, allowing you to reset things and get the history."""

    def __call__(self, command: str, verbose: bool = False):
        raise NotImplementedError("This method should be implemented by the subclass.")
