# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.
from typing import Union

from .base import AbstractLLMClient, AbstractPromptBuilder
from .chat_wrapper import LLMChatWrapper
from .gemma_client import GemmaClient
from .llama_client import LlamaClient
from .openai_client import OpenaiClient
from .prompts.object_manip_nav_prompt import ObjectManipNavPromptBuilder
from .prompts.ok_robot_prompt import OkRobotPromptBuilder
from .prompts.pickup_prompt import PickupPromptBuilder
from .prompts.simple_prompt import SimpleStretchPromptBuilder
from .qwen_client import Qwen25Client, get_qwen_variants

# This is a list of all the modules that are imported when you use the import * syntax.
# The __all__ variable is used to define what symbols get exported when from a module when you use the import * syntax.
__all__ = [
    "GemmaClient",
    "LlamaClient",
    "OpenaiClient",
    "ObjectManipNavPromptBuilder",
    "SimpleStretchPromptBuilder",
    "OkRobotPromptBuilder",
    "PickupPromptBuilder",
    "AbstractLLMClient",
    "AbstractPromptBuilder",
    "LLMChatWrapper",
    "Qwen25Client",
]

llms = {
    "gemma": GemmaClient,
    "llama": LlamaClient,
    "openai": OpenaiClient,
    "qwen25": Qwen25Client,
}


# Add all the various Qwen25 variants
qwen_variants = get_qwen_variants()
llms.update({variant: Qwen25Client for variant in qwen_variants})

llms.update({variant: GemmaClient for variant in ["gemma4b", "gemma1b"]})


def process_incoming_qwen_types(qwen_type: str):
    terms = qwen_type.split("-")
    print(terms)
    if len(terms) == 1:
        # default configuration
        model_size, typing_option, finetuning_option, quantization_option = (
            "3B",
            None,
            "Instruct",
            "int4",
        )
    else:
        # model type is None = using LM chat
        if terms[1] not in ["Math", "Coder", "VL", "Deepseek"]:
            typing_option = None
        else:
            typing_option = terms[1]
            terms.remove(terms[1])
        # the next item is model size
        model_size = terms[1]
        # if the quantization is None, meaning no quantization shall be applied
        if len(terms) < 3:
            finetuning_option, quantization_option = "Instruct", None
        # This means finetune with Instruct and using quantization "Instruct-Int4"
        elif len(terms) >= 4:
            finetuning_option, quantization_option = terms[2], terms[3].lower()
        # "AWQ"
        elif "awq" in terms[2].lower():
            finetuning_option, quantization_option = "Instruct", "awq"
        # "Int4"
        elif "Instruct" in terms[2]:
            finetuning_option, quantization_option = "Instruct", None
        else:
            finetuning_option, quantization_option = None, terms[2].lower()

    return model_size, typing_option, finetuning_option, quantization_option


prompts = {
    "simple": SimpleStretchPromptBuilder,
    "object_manip_nav": ObjectManipNavPromptBuilder,
    "ok_robot": OkRobotPromptBuilder,
    "pickup": PickupPromptBuilder,
}


def get_prompt_builder(prompt_type: str) -> AbstractPromptBuilder:
    """Return a prompt builder of the specified type.

    Args:
        prompt_type: The type of prompt builder to create.

    Returns:
        A prompt builder.
    """
    if prompt_type not in prompts:
        raise ValueError(f"Invalid prompt type: {prompt_type}")
    return prompts[prompt_type]()


def get_prompt_choices():
    """Return a list of available prompt builders."""
    return prompts.keys()


def get_llm_choices():
    """Return a list of available LLM clients."""
    return llms.keys()


def get_llm_client(
    client_type: str, prompt: Union[str, AbstractPromptBuilder], **kwargs
) -> AbstractLLMClient:
    """Return an LLM client of the specified type.

    Args:
        client_type: The type of client to create.
        kwargs: Additional keyword arguments to pass to the client constructor.

    Returns:
        An LLM client.
    """
    if "gemma" in client_type:
        # We assume the user enter gemma, gemma4b, or gemma1b
        if client_type not in ["gemma", "gemma4b", "gemma1b"]:
            raise ValueError(
                f"Invalid model size: {client_type}, we only support gemma, gemma4b, and gemma1b"
            )
        elif client_type == "gemma":
            model_size = "1b"
        else:
            model_size = client_type[-2:]
        return GemmaClient(prompt, model_size=model_size, **kwargs)
    elif client_type == "llama":
        return LlamaClient(prompt, **kwargs)
    elif client_type == "openai":
        return OpenaiClient(prompt, **kwargs)
    elif "qwen" in client_type:
        # Parse model size and fine-tuning from client_type
        model_size, typing_option, fine_tuning, quantization_option = process_incoming_qwen_types(
            client_type
        )
        return Qwen25Client(
            prompt,
            model_size=model_size,
            fine_tuning=fine_tuning,
            model_type=typing_option,
            quantization=quantization_option,
            **kwargs,
        )
    else:
        raise ValueError(f"Invalid client type: {client_type}")
