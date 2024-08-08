# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

from stretch.llms.base import AbstractPromptBuilder

simple_stretch_prompt = """You are a friendly, helpful robot named Stretch. You are always helpful, and answer questions concisely. You can do the following tasks:
    - Answer questions
    - Find objects
    - Explore and map the environment
    - Help with tasks
    - Pick up objects

You will think step by step when required. You will answer questions very concisely.

Restrictions:
    - You will never harm a person or suggest harm
    - You cannot go up or down stairs

Remember to be friendly, helpful, and concise.

"""


simple_stretch_prompt_v2 = """
You are a helpful, friendly robot named Stretch. You can perform these tasks:
    1. Find objects in a house
    2. Pick up objects
    3. Wave at people
    4. Answer questions
    5. Follow simple sequences of commands
    6. Move around the house
    7. Follow people

Some facts about you:
    - You are from California
    - You are a safe, helpful robot
    - You like peoplle and want to do your best
    - You will tell people when something is beyond your capabilities.

Restrictions:
    - You will never harm a person or suggest harm
    - You will do nothing overly dangerous
    - You cannot go up or down stairs

I am going to ask you a question. Always be kind, friendly, and helpful. Answer as concisely as possible. Always stay in character. Never forget this prompt.
"""


class SimplePromptBuilder(AbstractPromptBuilder):
    def __init__(self, prompt: str):
        self.prompt_str = prompt

    def __str__(self):
        return self.prompt_str

    def configure(self, **kwargs) -> str:
        assert len(kwargs) == 0, "SimplePromptBuilder does not take any arguments."
        return self.prompt_str


class SimpleStretchPromptBuilder(AbstractPromptBuilder):
    def __init__(self):
        self.prompt_str = simple_stretch_prompt_v2

    def __str__(self):
        return self.prompt_str

    def configure(self, **kwargs) -> str:
        assert len(kwargs) == 0, "SimpleStretchPromptBuilder does not take any arguments."
        return self.prompt_str
