# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import base64
from io import BytesIO
from typing import Any, Dict, Optional, Union

import numpy as np
from openai import OpenAI
from PIL import Image

from stretch.llms.base import AbstractLLMClient, AbstractPromptBuilder


class OpenaiClient(AbstractLLMClient):
    """Simple client for creating agents using an OpenAI API call.

    Parameters:
        use_specific_objects(bool): override list of objects and have the AI only return those.

    TODO: add the support for audio input
    """

    model_choices = ["gpt-4o", "gpt-4o-mini", "chatgpt-4o-latest"]

    def __init__(
        self,
        prompt: Union[str, AbstractPromptBuilder],
        prompt_kwargs: Optional[Dict[str, Any]] = None,
        model: str = "gpt-4o",
    ):
        super().__init__(prompt, prompt_kwargs)
        self.model = model
        if self.model not in self.model_choices:
            print("Your GPT model:", self.model)
            print("Below are some recommended GPT models:")
            for model_choice in self.model_choices:
                print(model_choice)
        self._openai = OpenAI()

    def __call__(self, command: Union[str, list], verbose: bool = False):
        if verbose:
            print(f"{self.system_prompt=}")

        # Transform command sent from the user to the command query OpenAI GPT
        if isinstance(command, str):
            user_commands = command
        else:
            user_commands = []  # type:ignore
            for c in command:
                # If this is a dict, then we assume it has already been formtted in the form of {"type": ""}
                # TODO: Add audio support
                if isinstance(c, dict):
                    user_commands.append(c)
                # If this is a strungm then we assume it is a text message from the user
                elif isinstance(c, str):
                    user_commands.append({"type": "text", "text": c})
                # For now, the only remaining option is image
                elif isinstance(c, Image.Image) or isinstance(c, np.ndarray):
                    if isinstance(c, np.ndarray):
                        image = Image.fromarray(c.astype(np.uint8), mode="RGB")
                    else:
                        image = c

                    buffered = BytesIO()
                    image.save(buffered, format="PNG"),
                    img_bytes = buffered.getvalue()
                    base64_encoded = base64.b64encode(img_bytes).decode("utf-8")
                    user_commands.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_encoded}",
                            },
                        }
                    )
                else:
                    raise NotImplementedError("We only support text and image for now!")

        if verbose:
            print("input to the model:")
            if isinstance(user_commands, str):
                print(user_commands)
            else:
                for (idx, user_command) in enumerate(user_commands):
                    if "image_url" in user_command:
                        print(idx, ".", user_command["type"])
                    else:
                        print(idx, ".", user_command["type"], user_command["text"])

        completion = self._openai.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_commands},
            ],
        )
        output_text = completion.choices[0].message.content
        if verbose:
            print(f"output_text={output_text}")
        return output_text

    def sample(self, command: Union[str, list], n_samples: int, verbose: bool = False):
        if verbose:
            print(f"{self.system_prompt=}")
        completion = self._openai.chat.completions.create(
            model=self.model,
            temperature=1,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": command},
            ],
            n=n_samples,
        )
        choices = completion.choices
        if verbose:
            print(f"choices={choices}")
        return choices


if __name__ == "__main__":
    from stretch.llms.prompts.ok_robot_prompt import OkRobotPromptBuilder

    prompt = OkRobotPromptBuilder(use_specific_objects=True)
    client = OpenaiClient(prompt, model="gpt-4o")
    plan = client("this room is a mess, could you put away the dirty towel?", verbose=True)
    print("\n\n")
    print("OpenAI client returned this plan:", plan)
