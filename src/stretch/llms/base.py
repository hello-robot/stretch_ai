# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image


class AbstractPromptBuilder(ABC):
    """Abstract base class for a prompt generator."""

    def __init__(self, **kwargs):
        print(" kwargs: ")
        print(kwargs)
        self.prompt_str = self.configure(**kwargs)

    def configure(self, **kwargs) -> str:
        """Configure the prompt with the given parameters, then return the prompt string."""

    def __str__(self) -> str:
        """Return the system prompt string for an LLM."""
        return self.prompt_str

    def __call__(self, kwargs: Optional[Dict[str, Any]] = None) -> str:
        """Return the system prompt string for an LLM."""
        if kwargs is not None:
            self.prompt_str = self.configure(**kwargs)
        return self.prompt_str

    def parse_response(self, response: str) -> Any:
        """Parse the response from the LLM. Usually does nothing."""
        return response

    def get_available_actions(self) -> List[str]:
        """Return a list of available actions."""
        return []


class AbstractLLMClient(ABC):
    """Abstract base class for a client that interacts with a language model."""

    def __init__(
        self,
        prompt: Union[str, AbstractPromptBuilder],
        prompt_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.prompt_kwargs = prompt_kwargs
        self.reset()

        # If the prompt is a string, use it as the prompt. Otherwise, generate the prompt string.
        if prompt is None:
            self._prompt = ""
        elif isinstance(prompt, str):
            self._prompt = prompt
        else:
            if prompt_kwargs is None:
                prompt_kwargs = {}
            self._prompt = prompt(**prompt_kwargs)

    @property
    def system_prompt(self) -> str:
        """Return the system prompt string for an LLM."""
        return self._prompt

    def reset(self) -> None:
        """Reset the client state."""
        self.conversation_history: List[Union[str, Dict[str, str]]] = []
        self._iterations = 0

    def is_first_message(self) -> bool:
        """Return True if the client has not yet sent a message."""
        return len(self.conversation_history) == 0

    @property
    def steps(self) -> int:
        """Return the number of steps taken by the client."""
        return self._iterations

    def add_history(self, message: Any) -> None:
        """Add a message to the conversation history."""
        self.conversation_history.append(message)

    def get_history(self) -> List[Any]:
        """Get the conversation history."""
        return self.conversation_history.copy()

    def get_history_as_str(self) -> str:
        """Return the conversation history as a string."""
        history = self.get_history()
        history_str = ""
        for item in history:
            if isinstance(item, str):
                history_str += item
            else:
                history_str += f"\n{item['role']}: {item['content']}"
        return history_str

    @abstractmethod
    def __call__(self, command: str, image: Optional[Image.Image] = None, verbose: bool = False):
        """Interact with the language model to generate a plan."""

    def parse(self, content: str) -> List[Tuple[str, str]]:
        """parse into list"""
        plan = []
        for command in content.split("\n"):
            action, target = command.split("=")
            plan.append((action, target))
        return plan
