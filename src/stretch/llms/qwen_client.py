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
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from stretch.llms.base import AbstractLLMClient, AbstractPromptBuilder

qwen_fine_tuning_options = ["Instruct", "Coder", "Math", "Deepseek"]
qwen_sizes = ["0.5B", "1.5B", "3B", "7B", "14B", "32B", "72B"]


class Qwen25Client(AbstractLLMClient):
    def __init__(
        self,
        prompt: Union[str, AbstractPromptBuilder],
        prompt_kwargs: Optional[Dict[str, Any]] = None,
        model_size: str = "3B",
        fine_tuning: str = "Instruct",
        max_tokens: int = 4096,
        device: str = "cuda",
        quantization: Optional[str] = "int4",
    ):
        super().__init__(prompt, prompt_kwargs)
        assert device in ["cuda", "mps"], f"Invalid device: {device}"
        assert model_size in qwen_sizes, f"Invalid model size: {model_size}"
        assert fine_tuning in qwen_fine_tuning_options, f"Invalid fine-tuning: {fine_tuning}"

        self.max_tokens = max_tokens
        if fine_tuning == "Deepseek":
            model_name = f"deepseek-ai/DeepSeek-R1-Distill-Qwen-{model_size}"
        else:
            model_name = f"Qwen/Qwen2.5-{model_size}-{fine_tuning}"

        print(f"Loading model: {model_name}")
        model_kwargs = {"torch_dtype": "auto"}

        quantization_config = None
        if quantization is not None:
            quantization = quantization.lower()
            # Note: there were supposed to be other options but this is the only one that worked this way
            if quantization == "awq":
                model_kwargs["torch_dtype"] = torch.float16
                model_name += "-AWQ"
            elif quantization in ["int8", "int4"]:
                try:
                    import bitsandbytes  # noqa: F401
                    from transformers import BitsAndBytesConfig, pipeline
                except ImportError:
                    raise ImportError(
                        "bitsandbytes required for int4/int8 quantization: pip install bitsandbytes"
                    )

                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=(quantization == "int4"),
                    load_in_8bit=(quantization == "int8"),
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                model_kwargs["quantization_config"] = quantization_config
            else:
                raise ValueError(f"Unknown quantization method: {quantization}")

        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
        )
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=device,
            model_kwargs=model_kwargs,
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
