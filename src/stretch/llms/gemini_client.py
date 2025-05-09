# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import os
from typing import Any, Dict, Optional, Union

from google import genai

from stretch.llms.base import AbstractLLMClient, AbstractPromptBuilder


class GeminiClient(AbstractLLMClient):
    """Simple client for creating agents using an OpenAI API call.

    Parameters:
        use_specific_objects(bool): override list of objects and have the AI only return those."""

    model_choices = [
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "gemini-2.5-pro-preview-03-25",
        "gemini-2.5-flash-preview-04-17",
    ]

    def __init__(
        self,
        prompt: Union[str, AbstractPromptBuilder],
        prompt_kwargs: Optional[Dict[str, Any]] = None,
        model: str = "gemini-2.0-flash",
    ):
        super().__init__(prompt, prompt_kwargs)
        self.model = model
        assert (
            self.model in self.model_choices
        ), f"model must be one of {self.model_choices}, got {self.model}"

        if "GOOGLE_API_KEY" in os.environ:
            self._gemini = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        else:
            raise Exception("Gemini token has not been set up yet!")

    def __call__(
        self, command: Union[str, list], model: Optional[str] = None, verbose: bool = False
    ):
        # prompt = copy.copy(self.prompt)
        # prompt = prompt.replace("$COMMAND", command)
        if verbose:
            print(f"{self.system_prompt=}")
        if model is None:
            model = self.model
        if isinstance(command, str):
            command = [command]
        command = [self.system_prompt] + command
        response = self._gemini.models.generate_content(model=self.model, contents=command)

        plan = response.text
        if verbose:
            print(f"plan={plan}")
        return plan

    def sample(
        self,
        command: Union[str, list],
        model: Optional[str] = None,
        n_samples: int = 4,
        verbose: bool = False,
    ):
        if verbose:
            print(f"{self.system_prompt=}")
        plan = []
        for _ in range(n_samples):
            plan.append(self.__call__(command, model=model, verbose=verbose))
        if verbose:
            print(f"plan={plan}")
        return plan
