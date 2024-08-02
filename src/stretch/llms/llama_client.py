from stretch.llms.base import AbstractLLMClient, AbstractPrompt


class LlamaClient(AbstractLLMClient):
    def __call__(self, command: str, verbose: bool = False):
        raise NotImplementedError("This method should be implemented by the subclass.")
