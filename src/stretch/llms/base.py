from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union


class Prompt(ABC):
    """Abstract base class for a prompt generator."""

    def __init__(self, **kwargs):
        self.prompt_str = self.configure(**kwargs)

    @abstractmethod
    def configure(self, **kwargs) -> str:
        """Configure the prompt with the given parameters, then return the prompt string."""
        pass

    def __call__(self, kwargs: Optional[Dict[str, Any]] = None) -> str:
        """Return the system prompt string for an LLM."""
        if kwargs is not None:
            self.prompt_str = self.configure(**kwargs)
        return self.prompt_str


class AbstractLLMClient(ABC):
    """Abstract base class for a client that interacts with a language model."""

    def __init__(self, prompt: Union[str, Prompt], prompt_kwargs: Optional[Dict[str, Any]] = None):
        self.prompt_kwargs = prompt
        self.reset()

        # If the prompt is a string, use it as the prompt. Otherwise, generate the prompt string.
        if isinstance(prompt, str):
            self.prompt = prompt
        else:
            self.prompt = prompt(**prompt_kwargs)

    def reset(self) -> None:
        """Reset the client state."""
        self.conversation_history: List[Dict[str, Any]] = []
        self.iterations = 0

    @abstractmethod
    def __call__(self, command: str, verbose: bool = False):
        """Interact with the language model to generate a plan."""
        pass

    def parse(self, content: str) -> List[Tuple[str, str]]:
        """parse into list"""
        plan = []
        for command in content.split("\n"):
            action, target = command.split("=")
            plan.append((action, target))
        return plan
