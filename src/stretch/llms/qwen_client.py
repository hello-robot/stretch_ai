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

from termcolor import colored
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from stretch.llms.base import AbstractLLMClient, AbstractPromptBuilder


class Qwen25Client(AbstractLLMClient):
    def __init__(
        self,
        prompt: Union[str, AbstractPromptBuilder],
        prompt_kwargs: Optional[Dict[str, Any]] = None,
        model_size: str = "3B",
        fine_tuning: str = "Instruct",
        max_tokens: int = 4096,
        device: str = "cuda",
    ):
        super().__init__(prompt, prompt_kwargs)
        assert device in ["cuda", "mps"], f"Invalid device: {device}"
        assert model_size in [
            "0.5B",
            "1.5B",
            "3B",
            "7B",
            "14B",
            "32B",
            "72B",
        ], f"Invalid model size: {model_size}"
        assert fine_tuning in ["Instruct", "Coder", "Math"], f"Invalid fine-tuning: {fine_tuning}"

        self.max_tokens = max_tokens
        model_name = f"Qwen/Qwen2.5-{model_size}-{fine_tuning}"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
        )
        self.pipe = pipeline(
            "text-generation", model=self.model, tokenizer=self.tokenizer, device=device
        )

    def __call__(self, command: str, verbose: bool = False):
        if self.is_first_message():
            system_message = {"role": "system", "content": self.system_prompt}
            self.add_history(system_message)

        # Prepare the messages including the conversation history
        new_message = {"role": "user", "content": command}

        self.add_history(new_message)
        messages = self.get_history()

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        t0 = timeit.default_timer()
        outputs = self.pipe(text, max_new_tokens=self.max_tokens)
        t1 = timeit.default_timer()

        assistant_response = outputs[0]["generated_text"].split("assistant")[-1].strip()

        self.add_history({"role": "assistant", "content": assistant_response})
        if verbose:
            print(f"Assistant response: {assistant_response}")
            print(f"Time taken: {t1 - t0:.2f}s")
        return assistant_response


if __name__ == "__main__":
    # from stretch.llms.prompts.object_manip_nav_prompt import ObjectManipNavPromptBuilder
    from stretch.llms.prompts.pickup_prompt import PickupPromptBuilder

    prompt = PickupPromptBuilder()
    client = Qwen25Client(prompt, model_size="1.5B", fine_tuning="Instruct")
    for _ in range(50):
        msg = input("Enter a message (empty to quit): ")
        if len(msg) == 0:
            break
        response = client(msg, verbose=True)
        print()
        print("-" * 80)
        print(colored("You said:", "green"), msg)
        print(colored("Response", "blue"), response)
