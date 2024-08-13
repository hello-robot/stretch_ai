# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import copy
from typing import Optional

from typing_extensions import override

from stretch.llms.base import AbstractPromptBuilder

DEFAULT_OBJECTS = "fanta can, tennis ball, black head band, purple shampoo bottle, toothpaste, orange packaging, green hair cream jar, green detergent pack,  blue moisturizer, green plastic cover, storage container, blue hair oil bottle, blue pretzels pack, blue hair gel tube, red bottle, blue bottle,  wallet"

DEFAULT_LOCATIONS = "white table, chair, dustbin, gray bed"

PROMPT_INTRO = """
Convert a command into formatted text using some combination of the following two commands:

pick=obj
place=loc

where obj can be a name describing some common household object such that it can be detected by an open-vocabulary object detector, and loc can be some household location which can be detected in the same way.

"""

PROMPT_SPECIFICS = """
obj may be any of these, or something specified in the command: $OBJECTS

loc may be any of these, or something specified in the command: $LOCATIONS
"""

PROMPT_EXAMPLES = """
Example 1:
Command: "get rid of that dirty towel"
Returns:
pick=towel
place=basket

Example 2:
Command: "put the cup in the sink"
Returns:
pick=cup
place=sink

Example 3:
Command: "i need the yellow shampoo bottle, can you put it by the shower?"
Returns:
pick=yellow bottle
place=bathroom counter

Example 4:
Command: "i could really use a sugary drink, i'm going to go lie down"
Returns:
pick=fanta can
place=gray bed

Example 5:
Command: "put the apple and orange on the kitchen table."
Returns:
pick=apple
place=kitchen table
pick=orange
place=kitchen table

You will respond ONLY with the executable commands, i.e. the part following "Returns." Do not include the word Returns. Objects must be specific. The term on the left side of the equals sign must be either pick or place.
"""


class OkRobotPromptBuilder(AbstractPromptBuilder):
    @override
    def configure(
        self,
        objects: Optional[str] = None,
        locations: Optional[str] = None,
        use_specific_objects: bool = True,
    ):

        self.use_specific_objects = use_specific_objects
        if objects is None:
            objects = DEFAULT_OBJECTS
        if locations is None:
            locations = DEFAULT_LOCATIONS
        self.objects = objects
        self.locations = locations
        if self.use_specific_objects:
            specifics = copy.copy(PROMPT_SPECIFICS)
            specifics = specifics.replace("$OBJECTS", self.objects)
            specifics = specifics.replace("$LOCATIONS", self.locations)
            prompt_str = PROMPT_INTRO + specifics + PROMPT_EXAMPLES
        else:
            prompt_str = PROMPT_INTRO + PROMPT_EXAMPLES
        return prompt_str
