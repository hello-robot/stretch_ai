# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import pytest

from stretch.llms.base import AbstractPromptBuilder
from stretch.llms.prompts import (
    ObjectManipNavPromptBuilder,
    OkRobotPromptBuilder,
    SimpleStretchPromptBuilder,
)

prompt_builders = [
    OkRobotPromptBuilder(use_specific_objects=True),
    OkRobotPromptBuilder(use_specific_objects=False),
    SimpleStretchPromptBuilder(),
    ObjectManipNavPromptBuilder(),
]


@pytest.mark.parametrize("prompt_builder", prompt_builders)
def test_prompt_builder(prompt_builder):
    assert isinstance(prompt_builder, AbstractPromptBuilder)
    assert isinstance(prompt_builder.prompt_str, str)
    assert isinstance(prompt_builder.configure(), str)
    assert isinstance(prompt_builder.__str__(), str)
    assert isinstance(prompt_builder.__call__(), str)
    assert isinstance(prompt_builder.__call__({}), str)


if __name__ == "__main__":
    for prompt_builder in prompt_builders:
        print(f"Testing {type(prompt_builder)} = {prompt_builder}")
        test_prompt_builder(prompt_builder)
