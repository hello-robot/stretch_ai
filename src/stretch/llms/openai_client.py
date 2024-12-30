# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

from typing import Any, Dict, Optional, Union

from openai import OpenAI

from stretch.llms.base import AbstractLLMClient, AbstractPromptBuilder


class OpenaiClient(AbstractLLMClient):
    """Simple client for creating agents using an OpenAI API call.

    Parameters:
        use_specific_objects(bool): override list of objects and have the AI only return those."""

    model_choices = [
        "gpt-3.5-turbo",
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4o-2024-11-20",
        "gpt-4o-2024-05-13",
    ]

    def __init__(
        self,
        prompt: Union[str, AbstractPromptBuilder],
        prompt_kwargs: Optional[Dict[str, Any]] = None,
        model: str = "gpt-4o",
    ):
        super().__init__(prompt, prompt_kwargs)
        self.model = model
        assert (
            self.model in self.model_choices
        ), f"model must be one of {self.model_choices}, got {self.model}"
        self._openai = OpenAI()

    def __call__(self, command: Union[str, list], verbose: bool = False):
        # prompt = copy.copy(self.prompt)
        # prompt = prompt.replace("$COMMAND", command)
        if verbose:
            print(f"{self.system_prompt=}")
        completion = self._openai.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": command},
            ],
        )
        plan = completion.choices[0].message.content
        if verbose:
            print(f"plan={plan}")
        return plan


if __name__ == "__main__":
    from stretch.llms.prompts.ok_robot_prompt import OkRobotPromptBuilder

    prompt = OkRobotPromptBuilder(use_specific_objects=True)
    # client = OpenaiClient(prompt, model="gpt-4o-mini")
    # client = OpenaiClient(prompt, model="gpt-3.5-turbo")
    client = OpenaiClient(prompt, model="gpt-4o")
    plan = client("this room is a mess, could you put away the dirty towel?", verbose=True)
    print("\n\n")
    print("OpenAI client returned this plan:", plan)
