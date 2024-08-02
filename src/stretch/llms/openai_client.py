import copy
from typing import Any, Dict, List, Optional, Tuple, Union

from openai import OpenAI

from stretch.llms.base import AbstractLLMClient, AbstractPrompt


class OpenaiClient(AbstractLLMClient):
    """Simple client for creating agents using an OpenAI API call.

    Parameters:
        use_specific_objects(bool): override list of objects and have the AI only return those."""

    def __init__(
        self,
        prompt: Union[str, AbstractPrompt],
        prompt_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if isinstance(prompt, str):
            self.prompt = prompt
        else:
            self.prompt = prompt(**prompt_kwargs)
        self._openai = OpenAI()

    def __call__(self, command: str, verbose: bool = False):
        # prompt = copy.copy(self.prompt)
        # prompt = prompt.replace("$COMMAND", command)
        if verbose:
            print(f"{self.prompt=}")
        completion = self._openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": command},
            ],
        )
        plan = self.parse(completion.choices[0].message.content)
        if verbose:
            print(completion.choices[0].message)
            print(f"{plan=}")
        return plan


if __name__ == "__main__":
    from stretch.llms.prompts.ok_robot_prompt import OkRobotPrompt

    prompt = OkRobotPrompt(use_specific_objects=True)
    client = OpenaiClient(prompt)
    plan = client("this room is a mess, could you put away the dirty towel?", verbose=True)
    print("\n\n")
    print("OpenAI client returned this plan:", plan)
