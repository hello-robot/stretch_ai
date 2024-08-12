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

    def __init__(
        self,
        prompt: Union[str, AbstractPromptBuilder],
        prompt_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(prompt, prompt_kwargs)
        self._openai = OpenAI()

    def __call__(self, command: str, verbose: bool = False):
        # prompt = copy.copy(self.prompt)
        # prompt = prompt.replace("$COMMAND", command)
        if verbose:
            print(f"{self.prompt=}")
        completion = self._openai.chat.completions.create(
            model="gpt-3.5-turbo",
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
    from stretch.llms.prompts.ok_robot_prompt import OkRobotPrompt

    prompt = OkRobotPrompt(use_specific_objects=True)
    client = OpenaiClient(prompt)
    plan = client("this room is a mess, could you put away the dirty towel?", verbose=True)
    print("\n\n")
    print("OpenAI client returned this plan:", plan)
