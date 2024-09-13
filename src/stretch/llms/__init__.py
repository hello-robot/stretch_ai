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
from .gemma_client import Gemma2bClient
from .llama_client import LlamaClient
from .openai_client import OpenaiClient
from .prompts.object_manip_nav_prompt import ObjectManipNavPromptBuilder
from .prompts.simple_prompt import SimpleStretchPromptBuilder

# This is a list of all the modules that are imported when you use the import * syntax.
# The __all__ variable is used to define what symbols get exported when from a module when you use the import * syntax.
__all__ = [
    "Gemma2bClient",
    "LlamaClient",
    "OpenaiClient",
    "ObjectManipNavPromptBuilder",
    "SimpleStretchPromptBuilder",
]

llms = {
    "gemma2b": Gemma2bClient,
    "llama": LlamaClient,
    "openai": OpenaiClient,
}


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
    else:
        raise ValueError(f"Invalid client type: {client_type}")
