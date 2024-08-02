from typing import Optional, Union

import torch
import transformers
from termcolor import colored

from stretch.llms.base import AbstractLLMClient, AbstractPromptBuilder

default_model_id = "meta-llama/Meta-Llama-3.1-8B"


class LlamaClient(AbstractLLMClient):
    def __init__(self, prompt: Optional[Union[str, AbstractPromptBuilder]], model_id: str = None):
        super().__init__(prompt)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.model = transformers.GPT2LMHeadModel.from_pretrained(model_id).to(self.device)
        # self.tokenizer = transformers.GPT2TokenizerFast.from_pretrained(model_id)
        if model_id is None:
            model_id = default_model_id
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )

    def __call__(self, command: str, verbose: bool = False):
        raise NotImplementedError("This method should be implemented by the subclass.")


if __name__ == "__main__":
    from stretch.llms.prompts.simple_prompt import SimpleStretchPromptBuilder

    prompt = SimpleStretchPromptBuilder()
    client = LlamaClient(prompt)
    for _ in range(50):
        msg = input("Enter a message (empty to quit):")
        if len(msg) == 0:
            break
        response = client(msg)
        print(colored("You said:", "green"), msg)
        print(colored("Response", "blue"), response)
