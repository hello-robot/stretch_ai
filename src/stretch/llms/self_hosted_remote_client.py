import zmq

from stretch.llms.base import AbstractLLMClient, AbstractPrompt


class SelfHostedRemoteClient(AbstractLLMClient):
    """Client which runs on your own machine, locally or remotely, in order to give you the GPU to run your own LLM.
    It communicates with the LLM via a ZMQ socket. It will send and receive dictionaries, allowing you to reset things and get the history."""

    def __call__(self, command: str, verbose: bool = False):
        raise NotImplementedError("This method should be implemented by the subclass.")
