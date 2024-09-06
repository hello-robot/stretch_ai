# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import pytest
from PIL import Image

from stretch.perception.captioners import get_captioner

captioners = ["blip", "git", "vit_gpt2"]

# With these two images from docs:
images = ["../docs/object.png", "../docs/receptacle.png"]


def test_get_captioner():
    captioner = get_captioner("blip", {})
    assert captioner is not None
    assert captioner.__class__.__name__ == "BlipCaptioner"

    captioner = get_captioner("git", {})
    assert captioner is not None
    assert captioner.__class__.__name__ == "GitCaptioner"

    captioner = get_captioner("moondream", {})
    assert captioner is not None
    assert captioner.__class__.__name__ == "MoondreamCaptioner"

    captioner = get_captioner("vit_gpt2", {})
    assert captioner is not None
    assert captioner.__class__.__name__ == "VitGPT2Captioner"

    with pytest.raises(ValueError):
        get_captioner("invalid_captioner", {})


@pytest.mark.parametrize("captioner_name", captioners)
def test_get_captioner_all(captioner_name):
    captioner = get_captioner(captioner_name, {})
    assert captioner is not None

    with pytest.raises(ValueError):
        get_captioner("invalid_captioner", {})

    # Test captioning on the two images
    for image_path in images:
        captioner = get_captioner(captioner_name, {})
        assert captioner is not None

        image = Image.open(image_path)
        caption = captioner.caption_image(image)
        assert caption is not None
        assert isinstance(caption, str)
        assert len(caption) > 0
        print(f"Caption: {caption}")


if __name__ == "__main__":
    test_get_captioner()
    test_get_captioner_all("blip")
    test_get_captioner_all("git")
    test_get_captioner_all("moondream")
    test_get_captioner_all("vit_gpt2")
    print("All tests passed!")
