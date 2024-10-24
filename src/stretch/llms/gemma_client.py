# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import timeit
from typing import Any, Dict, Optional, Union

import torch
from termcolor import colored
from transformers import pipeline

from stretch.llms.base import AbstractLLMClient, AbstractPromptBuilder


class Gemma2bClient(AbstractLLMClient):
    def __init__(
        self,
        prompt: Union[str, AbstractPromptBuilder],
        prompt_kwargs: Optional[Dict[str, Any]] = None,
        max_tokens: int = 4096,
        device: str = "cuda",
    ):
        super().__init__(prompt, prompt_kwargs)
        assert device in ["cuda", "mps"], f"Invalid device: {device}"
        self.max_tokens = max_tokens
        self.pipe = pipeline(
            "text-generation",
            model="google/gemma-2-2b-it",
            model_kwargs={"torch_dtype": torch.bfloat16},
            device=device,
        )

    def __call__(self, command: str, verbose: bool = False):
        if self.is_first_message():
            new_message = {"role": "user", "content": self.system_prompt + command}
        else:
            new_message = {"role": "user", "content": command}

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
    # from stretch.llms.prompts.object_manip_nav_prompt import ObjectManipNavPromptBuilder
    from stretch.llms.prompts.pickup_prompt import PickupPromptBuilder

    prompt = PickupPromptBuilder()

    # prompt = ObjectManipNavPromptBuilder()
    client = Gemma2bClient(prompt)
    for _ in range(50):
        msg = input("Enter a message (empty to quit): ")
        if len(msg) == 0:
            break
        response = client(msg)
        print(colored("You said:", "green"), msg)
        print(colored("Response", "blue"), response)
