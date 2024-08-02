import timeit
from typing import Any, Dict, Optional, Union

import torch
from termcolor import colored
from transformers import pipeline

from stretch.llms.base import AbstractLLMClient, AbstractPromptBuilder


class GemmaClient(AbstractLLMClient):
    def __init__(
        self,
        prompt: Union[str, AbstractPromptBuilder],
        prompt_kwargs: Optional[Dict[str, Any]] = None,
        max_tokens: int = 512,
    ):
        super().__init__(prompt, prompt_kwargs)
        self.max_tokens = max_tokens
        self.pipe = pipeline(
            "text-generation",
            model="google/gemma-2-2b-it",
            model_kwargs={"torch_dtype": torch.bfloat16},
            device="cuda",  # replace with "mps" to run on a Mac device
        )

    def __call__(self, command: str, verbose: bool = False):
        if self.steps == 0:
            new_message = {"role": "user", "content": msg}
        else:
            new_message = {"role": "user", "content": msg}

        self.add_history(new_message)
        # Prepare the messages including the conversation history
        messages = self.get_history()
        t0 = timeit.default_timer()
        outputs = self.pipe(
            messages, max_new_tokens=self.max_tokens, return_full=True, verbose=verbose
        )
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
    client = GemmaClient(prompt)
    for _ in range(50):
        msg = input("Enter a message (empty to quit):")
        if len(msg) == 0:
            break
        response = client(msg)
        print(colored("Response", "blue"), response)
