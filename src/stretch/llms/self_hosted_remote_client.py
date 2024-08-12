# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.


from stretch.llms.base import AbstractLLMClient


class SelfHostedRemoteClient(AbstractLLMClient):
    """Client which runs on your own machine, locally or remotely, in order to give you the GPU to run your own LLM.
    It communicates with the LLM via a ZMQ socket. It will send and receive dictionaries, allowing you to reset things and get the history."""

    def __call__(self, command: str, verbose: bool = False):
        raise NotImplementedError("This method should be implemented by the subclass.")
