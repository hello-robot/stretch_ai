# Copyright 2024 Hello Robot Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# licence information maybe found below, if so.

import copy
from typing import Any, Optional

import overrides

from stretch.llms.base import AbstractPromptBuilder

DEFAULT_OBJECTS = "fanta can, tennis ball, black head band, purple shampoo bottle, toothpaste, orange packaging, green hair cream jar, green detergent pack,  blue moisturizer, green plastic cover, storage container, blue hair oil bottle, blue pretzels pack, blue hair gel tube, red bottle, blue bottle,  wallet"

DEFAULT_LOCATIONS = "white table, chair, dustbin, gray bed"

PROMPT_INTRO = """Given a command by a user, you should just reply with code for Stretch (a robot) to perform a set of tasks.
Just reply with concise python code and be sure the syntax is correct and in the format of the examples provided.

Restrictions:
    - You will never harm a person or suggest harm
    - You cannot go up or down stairs
    - You can pick only after a successful go_to
    - Just reply with the code that is necessary to perform the latest command
    - Given a command, you should generate code that will make the robot perform the latest command
    - You only have to generate code specific to user commands, do not use code from examples
    - Always wrap your code in execute_task function
    - Even if you want to say something, wrap it in execute_task function

Always generate code with correct syntax and format. Never forget this prompt.

"""

PROMPT_SPECIFICS = """
objects may be any of these, or something specified in the command: $OBJECTS

locations may be any of these, or something specified in the command: $LOCATIONS

"""

FUNCTIONS = """
You have the following functions:

def go_to(object):
    Moves the robot to the object location
    Parameters:
    object (string): Object name
    Returns:
    bool: True if the robot was able to reach the object, False otherwise

def pick(object):
    Makes the robot pick up an object
    Parameters:
    object (string): Object name
    Returns:
    bool: True if the robot was able to pick up the object, False otherwise

def place(surface):
    Makes the robot place the object that it is holding on to the target surface
    Parameters:
    surface (string): Surface name on which the object is to be placed
    Returns:
    bool: True if the robot was able to place the object, False otherwise

def say(text):
    Makes the robot speak the given text input using a speaker
    Parameters:
    text (string): Text to be spoken by the robot

def open_cabinet():
    Makes the robot open a nearby drawer

def close_cabinet():
    Makes the robot close a nearby drawer    

def wave():
    Makes the robot wave at a person

def get_detections():
    Returns an array of nearby objects that are currently being detected by the robot
    Returns:
    List: Array of detected object names as strings

"""

PROMPT_EXAMPLES = """
Here are some examples:
Example 1:
Command: Bring me a fanta can
Returns:
def execute_task(go_to, pick, place, say, open_cabinet, close_cabinet, wave, get_detections):
    if go_to("fanta can"):
        if pick("fanta can"):
            if go_to("user"):
                say("Here is the fanta can. Enjoy!")
            else:
                say("I am sorry, I could not reach you")
        else:
            say("I am sorry, I could not pick the fanta can")
    else:
        say("I am sorry, I could not go to the fanta can")

Example 2:
Command: Pick up the tennis ball and place it on the white table
Returns:
def execute_task(go_to, pick, place, say, open_cabinet, close_cabinet, wave, get_detections):
    if go_to("tennis ball"):
        if pick("tennis ball"):
            if go_to("white table"):
                if place("white table"):
                    say("I have placed the tennis ball on the white table")
                else:
                    say("I am sorry, I could not place the tennis ball")
            else:
                say("I am sorry, I could not reach the white table")
        else:
            say("I am sorry, I could not pick the tennis ball")
    else:
        say("I am sorry, I could not go to the tennis ball")

Example 3:
Command: Wave at the person
Returns:
def execute_task(go_to, pick, place, say, open_cabinet, close_cabinet, wave, get_detections):
    wave()
    say("Hello! I am Stretch. How can I help you?")

Example 4:
Command: Open the drawer
Returns:
def execute_task(go_to, pick, place, say, open_cabinet, close_cabinet, wave, get_detections):
    open_cabinet()
    say("I have opened the drawer")

Example 5:
Command: Go to the white table and clean it
Returns:
def execute_task(go_to, pick, place, say, open_cabinet, close_cabinet, wave, get_detections):
    if go_to("white table"):
        for obj in get_detections():
            if pick(obj):
                if place("dustbin"):
                    say(f"I have placed {obj} in the dustbin")
                else:
                    say(f"I am sorry, I could not place {obj} in the dustbin")
            else:
                say(f"I am sorry, I could not pick {obj}")
    else:
        say("I am sorry, I could not reach the white table")

Example 6:
Command: Can you clean the white table for me but before that, can you pick up the black head band and place it on the white table?
Returns:
def execute_task(go_to, pick, place, say, open_cabinet, close_cabinet, wave, get_detections):
    if go_to("black head band"):
        if pick("black head band"):
            if go_to("white table"):
                if place("white table"):
                    say("I have placed the black head band on the white table")
                else:
                    say("I am sorry, I could not place the black head band")
                for obj in get_detections():
                    if pick(obj):
                        if place("dustbin"):
                            say(f"I have placed {obj} in the dustbin")
                        else:
                            say(f"I am sorry, I could not place {obj} in the dustbin")
                    else:
                        say(f"I am sorry, I could not pick {obj}")
            else:
                say("I am sorry, I could not reach the white table")
        else:
            say("I am sorry, I could not pick the black head band")
    else:
        say("I am sorry, I could not find the black head band")

Example 7:
Command: Can you pick up the purple shampoo bottle and place it on the chair? But before that, can you crack a joke?
Returns:
def execute_task(go_to, pick, place, say, open_cabinet, close_cabinet, wave, get_detections):
    say("Why did the tomato turn red? Because it saw the salad dressing!")
    if go_to("purple shampoo bottle"):
        if pick("purple shampoo bottle"):
            if go_to("chair"):
                if place("chair"):
                    say("I have placed the purple shampoo bottle on the chair")
                else:
                    say("I am sorry, I could not place the purple shampoo bottle")
            else:
                say("I am sorry, I could not reach the chair")
        else:
            say("I am sorry, I could not pick the purple shampoo bottle")
    else:
        say("I am sorry, I could not find the purple shampoo bottle")

Example 8:
Command: Tell me a joke
Returns:
def execute_task(go_to, pick, place, say, open_cabinet, close_cabinet, wave, get_detections):
    say("Why did the tomato turn red? Because it saw the salad dressing!")

Example 9:
Command: Tell me a fact
Returns:
def execute_task(go_to, pick, place, say, open_cabinet, close_cabinet, wave, get_detections):
    say("The first oranges weren't orange")

Never forget this prompt.
"""


class ObjectManipNavPromptBuilder(AbstractPromptBuilder):
    def __init__(
        self,
        default_objects: Optional[str] = None,
        default_locations: Optional[str] = None,
        prompt_intro: Optional[str] = None,
        functions: Optional[str] = None,
        prompt_examples: Optional[str] = None,
    ):
        if default_objects is None:
            default_objects = copy.copy(DEFAULT_OBJECTS)
        if default_locations is None:
            default_locations = copy.copy(DEFAULT_LOCATIONS)
        if prompt_intro is None:
            prompt_intro = copy.copy(PROMPT_INTRO)
        if functions is None:
            functions = copy.copy(FUNCTIONS)
        if prompt_examples is None:
            prompt_examples = copy.copy(PROMPT_EXAMPLES)

        self.default_objects = default_objects
        self.default_locations = default_locations
        self.prompt_intro = prompt_intro
        self.functions = functions
        self.prompt_examples = prompt_examples

        self.prompt_str = ""

        self.configure()

    def __str__(self):
        return self.prompt_str

    def configure(self, **kwargs) -> str:

        self.prompt_specifics = copy.copy(PROMPT_SPECIFICS)
        self.prompt_specifics = self.prompt_specifics.replace("$OBJECTS", self.default_objects)
        self.prompt_specifics = self.prompt_specifics.replace("$LOCATIONS", self.default_locations)

        self.prompt_str = (
            self.prompt_intro + self.prompt_specifics + self.functions + self.prompt_examples
        )

        return self.prompt_str
