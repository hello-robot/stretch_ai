from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union


class AbstractPromptBuilder(ABC):
    """Abstract base class for a prompt generator."""

    def __init__(self, **kwargs):
        print(" kwargs: ")
        print(kwargs)
        self.prompt_str = self.configure(**kwargs)

    @abstractmethod
    def configure(self, **kwargs) -> str:
        """Configure the prompt with the given parameters, then return the prompt string."""
        pass

    def __str__(self) -> str:
        """Return the system prompt string for an LLM."""
        return self.prompt_str

    def __call__(self, kwargs: Optional[Dict[str, Any]] = None) -> str:
        """Return the system prompt string for an LLM."""
        if kwargs is not None:
            self.prompt_str = self.configure(**kwargs)
        return self.prompt_str


class AbstractLLMClient(ABC):
    """Abstract base class for a client that interacts with a language model."""

    def __init__(
        self,
        prompt: Union[str, AbstractPromptBuilder],
        prompt_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.prompt_kwargs = prompt
        self.reset()

        # If the prompt is a string, use it as the prompt. Otherwise, generate the prompt string.
        if isinstance(prompt, str):
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
        self.conversation_history: List[Dict[str, Any]] = []
        self._iterations = 0

    @property
    def steps(self) -> int:
        """Return the number of steps taken by the client."""
        return self._iterations

    def add_history(self, message: Dict[str, Any]) -> None:
        """Add a message to the conversation history."""
        self.conversation_history.append(message)

    def get_history(self) -> List[Dict[str, Any]]:
        """Get the conversation history."""
        return self.conversation_history.copy()

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
