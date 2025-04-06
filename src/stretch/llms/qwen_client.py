# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import timeit
from typing import Any, Dict, List, Optional, Union

import torch
from PIL import Image
from termcolor import colored
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from stretch.llms.base import AbstractLLMClient, AbstractPromptBuilder

# Coder: 32B, 14B, 7B, 3B, 1.5B, 0.5B, (None, Instruct, Instruct-AWQ, Instruct-GGUF, Instruct-GPTQ-Int4, Instruct-GPTQ-Int8)
# Math: 72B, 7B, 1.5B (None, Instruct)
# "0.5B", "1.5B", "3B", "7B", "14B", "32B", "72B"
# Deepseek: 1.5B, 7B, 14B, 32B

qwen_typing_options = ["Math", "Coder", "Deepseek", None]
qwen_quantization_options = {
    None: [None, "AWQ", "Int4", "Int8", "Instruct", "Instruct-Int4", "Instruct-Int8"],
    "Coder": [None, "AWQ", "Int4", "Int8", "Instruct", "Instruct-Int4", "Instruct-Int8"],
    "Math": [None, "Int4", "Int8", "Instruct", "Instruct-Int4", "Instruct-Int8"],
    "Deepseek": [None, "Int4", "Int8"],
}
qwen_sizes = {
    None: ["0.5B", "1.5B", "3B", "7B", "14B", "32B", "72B"],
    "Coder": ["0.5B", "1.5B", "3B", "7B", "14B", "32B"],
    "Math": ["1.5B", "7B", "72B"],
    "Deepseek": ["1.5B", "7B", "14B", "72B"],
}


def get_qwen_variants():
    qwen_variants = []
    for qwen_typing_option in qwen_typing_options:
        for qwen_quantization_option in qwen_quantization_options[qwen_typing_option]:
            for qwen_size in qwen_sizes[qwen_typing_option]:
                qwen_type = "qwen25"
                if qwen_typing_option is not None:
                    qwen_type += "-" + qwen_typing_option
                qwen_type += "-" + qwen_size
                if qwen_quantization_option is not None:
                    qwen_type += "-" + qwen_quantization_option
                qwen_variants.append(qwen_type)
    return qwen_variants


class Qwen25Client(AbstractLLMClient):
    def __init__(
        self,
        prompt: Union[str, AbstractPromptBuilder],
        prompt_kwargs: Optional[Dict[str, Any]] = None,
        model_size: str = "3B",
        fine_tuning: Optional[str] = "Instruct",
        model_type: Optional[str] = None,
        max_tokens: int = 4096,
        device: str = "cuda",
        quantization: Optional[str] = "int4",
    ):
        super().__init__(prompt, prompt_kwargs)
        assert device in ["cuda", "mps"], f"Invalid device: {device}"
        assert model_type in qwen_typing_options, f"Invalid model type: {model_type}"
        assert model_size in qwen_sizes[model_type], f"Invalid model size: {model_size}"
        assert fine_tuning in [None, "Instruct"], f"Invalid fine-tuning: {fine_tuning}"

        self.max_tokens = max_tokens

        if model_type == "Deepseek":
            model_name = f"deepseek-ai/DeepSeek-R1-Distill-Qwen-{model_size}"
        elif model_type is None:
            if fine_tuning is None:
                model_name = f"Qwen/Qwen2.5-{model_size}"
            else:
                model_name = f"Qwen/Qwen2.5-{model_size}-{fine_tuning}"
        else:
            if fine_tuning is None:
                model_name = f"Qwen/Qwen2.5-{model_type}-{model_size}"
            else:
                model_name = f"Qwen/Qwen2.5-{model_type}-{model_size}-{fine_tuning}"

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
                    from transformers import BitsAndBytesConfig
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


from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


class Qwen25VLClient:
    def __init__(
        self,
        prompt: Optional[str] = None,
        model_size: str = "3B",
        fine_tuning: Optional[str] = "Instruct",
        max_tokens: int = 4096,
        num_beams: int = 1,
        device: str = "cuda",
        quantization: Optional[str] = "int4",
        use_fast_attn: bool = False,
    ):
        """
        A simple API for Qwen2.5-VL models

        Parameters:
            quantization: we support no quatization, AWQ, and bitsandbytes int4 and int8
        """
        self.system_prompt = prompt
        assert device in ["cuda", "mps"], f"Invalid device: {device}"
        assert model_size in ["3B", "7B", "72B"], f"Invalid model size: {model_size}"
        assert fine_tuning in [None, "Instruct"], f"Invalid fine-tuning: {fine_tuning}"

        self._device = device
        self.max_tokens = max_tokens
        self.num_beams = num_beams
        self.use_fast_attn = use_fast_attn

        if fine_tuning is None:
            model_name = f"Qwen/Qwen2.5-VL-{model_size}"
        else:
            model_name = f"Qwen/Qwen2.5-VL-{model_size}-{fine_tuning}"

        print(f"Loading model: {model_name}")
        model_kwargs = {"torch_dtype": "auto"}

        quantization_config = None
        if quantization is not None:
            quantization = quantization.lower()
            # Note: we only support AWQ and bitsandbytes here
            if quantization == "awq":
                model_kwargs["torch_dtype"] = torch.float16
                model_name += "-AWQ"
            elif quantization in ["int8", "int4"]:
                try:
                    import bitsandbytes  # noqa: F401
                    from transformers import BitsAndBytesConfig
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

        self.processor = AutoProcessor.from_pretrained(model_name)
        if self.use_fast_attn:
            attn_implementaion = "flash_attention_2"
        else:
            attn_implementaion = None
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            attn_implementation=attn_implementaion,
            device_map=device,
            **model_kwargs,
        )

    def __call__(
        self, command: Union[str, List[Dict[str, Any]], Image.Image], verbose: bool = False
    ):
        if self.system_prompt is not None:
            messages = [{"role": "system", "content": self.system_prompt}]
        else:
            messages = []

        # Prepare the messages
        if not isinstance(command, List):
            messages.append({"role": "user", "content": command})
        else:
            messages += command
        print(messages)

        t0 = timeit.default_timer()

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self._device)

        generated_ids = self.model.generate(
            **inputs, max_new_tokens=self.max_tokens, num_beams=self.num_beams
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        t1 = timeit.default_timer()

        if verbose:
            print(f"Assistant response: {output_text}")
            print(f"Time taken: {t1 - t0:.2f}s")

        return output_text


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
