import timeit
from typing import Optional, Union

import torch
import transformers
from termcolor import colored

from stretch.llms.base import AbstractLLMClient, AbstractPromptBuilder

default_model_id = "meta-llama/Meta-Llama-3.1-8B"


class LlamaClient(AbstractLLMClient):
    def __init__(
        self,
        prompt: Optional[Union[str, AbstractPromptBuilder]],
        model_id: str = None,
        max_tokens: int = 512,
    ):
        super().__init__(prompt)
        self.max_tokens = max_tokens
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.model = transformers.GPT2LMHeadModel.from_pretrained(model_id).to(self.device)
        # self.tokenizer = transformers.GPT2TokenizerFast.from_pretrained(model_id)
        if model_id is None:
            model_id = default_model_id
        self.pipe = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )

    def __call__(self, command: str, verbose: bool = False):
        if self.is_first_message():
            new_message = {"role": "user", "content": self.system_prompt + msg}
        else:
            new_message = {"role": "user", "content": msg}

        self.add_history(new_message)
        # Prepare the messages including the conversation history
        messages = self.get_history()
        t0 = timeit.default_timer()
        outputs = self.pipe(messages, max_new_tokens=self.max_tokens)
        t1 = timeit.default_timer()
        assistant_response = outputs[0]["generated_text"][-1]["content"].strip()

        # Add the assistant's response to the conversation history
        self.add_history({"role": "assistant", "content": assistant_response})
        if verbose:
            print(f"Assistant response: {assistant_response}")
            print(f"Time taken: {t1 - t0:.2f}s")
        return assistant_response


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
