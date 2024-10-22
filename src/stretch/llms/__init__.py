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
from .gemma_client import Gemma2bClient
from .llama_client import LlamaClient
from .openai_client import OpenaiClient
from .prompts.object_manip_nav_prompt import ObjectManipNavPromptBuilder
from .prompts.ok_robot_prompt import OkRobotPromptBuilder
from .prompts.pickup_prompt import PickupPromptBuilder
from .prompts.simple_prompt import SimpleStretchPromptBuilder
from .qwen_client import Qwen25Client

# This is a list of all the modules that are imported when you use the import * syntax.
# The __all__ variable is used to define what symbols get exported when from a module when you use the import * syntax.
__all__ = [
    "Gemma2bClient",
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
    "gemma2b": Gemma2bClient,
    "llama": LlamaClient,
    "openai": OpenaiClient,
    "qwen25": Qwen25Client,
}


# Add all the various Qwen25 variants
qwen_variants = []
for model_size in ["0.5B", "1.5B", "3B", "7B", "14B", "32B", "72B"]:
    for fine_tuning in ["Instruct", "Coder", "Math"]:
        qwen_variants.append(f"qwen25-{model_size}-{fine_tuning}")
        llms.update({variant: Qwen25Client for variant in qwen_variants})

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
    if client_type == "gemma2b":
        return Gemma2bClient(prompt, **kwargs)
    elif client_type == "llama":
        return LlamaClient(prompt, **kwargs)
    elif client_type == "openai":
        return OpenaiClient(prompt, **kwargs)
    elif "qwen" in client_type:
        # Parse model size and fine-tuning from client_type
        terms = client_type.split("-")
        if len(terms) == 3:
            model_size, fine_tuning = terms[1], terms[2]
        else:
            model_size, fine_tuning = "3B", "Instruct"
        return Qwen25Client(prompt, **kwargs)
    else:
        raise ValueError(f"Invalid client type: {client_type}")
