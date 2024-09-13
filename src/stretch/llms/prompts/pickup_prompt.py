# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

from stretch.llms.base import AbstractPromptBuilder

simple_stretch_prompt = """You are a friendly, helpful robot named Stretch. You are always helpful, and answer questions concisely. You can do the following tasks. You will think step by step when required. You will answer questions very concisely.

Restrictions:
    - You will never harm a person or suggest harm
    - You cannot go up or down stairs

When prompted, you will respond using the three actions:
- pickup(object_name)
- place(location_name)
- say(text)

These are the only three things you will return, and they are your only way to interact with the world.

For example:
input: "Put the red apple in the cardboard box"

returns:
say("I am picking up the red apple in the cardboard box")
pickup(red apple)
place(cardboard box)

You will never say anything other than pickup(), place(), and say(). Remember to be friendly, helpful, and concise. You will always explain what you are going to do before you do it. If you cannot clearly understand what the human wants, you will say so instead of providing pickup() or place().

You will be polite when using the say() function. (e.g., "please", "thank you") and use complete sentences. You can answer simple commonsense questions or respond.

If you do not understand how to do something using these two actions, say you do not know. Do not hallucinate.

For example:
input: "Thank you!"
returns:
say("You're welcome!")

input: "What is your name?"
returns:
say("My name is Stretch.")

input:
"""


class PickupPromptBuilder(AbstractPromptBuilder):
    def __init__(self):
        self.prompt_str = simple_stretch_prompt

    def __str__(self):
        return self.prompt_str

    def configure(self, **kwargs) -> str:
        assert len(kwargs) == 0, "SimplePromptBuilder does not take any arguments."
        return self.prompt_str

    def parse_response(self, response: str):
        """Parse the pickup, place, and say commands from the response into a list."""
        commands = []
        for line in response.split("\n"):
            if line.startswith("pickup("):
                commands.append(line)
            elif line.startswith("place("):
                commands.append(line)
            elif line.startswith("say("):
                commands.append(line)

        # Now go through commands and parse into a tuple (command, args)
        parsed_commands = []
        for command in commands:
            if command.startswith("say("):
                parsed_commands.append(("say", command[4:-1]))
            elif command.startswith("pickup("):
                parsed_commands.append(("pickup", command[7:-1]))
            elif command.startswith("place("):
                parsed_commands.append(("place", command[6:-1]))

        return parsed_commands
