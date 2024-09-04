# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.
import numpy as np
import pytest
import torch
from PIL import Image

images = ["../docs/object.png", "../docs/receptacle.png"]

from stretch.perception.encoders import encoders, get_encoder


def test_get_encoder():
    encoder = get_encoder("clip", {})
    assert encoder is not None
    assert encoder.__class__.__name__ == "ClipEncoder"

    encoder = get_encoder("normalized_clip", {})
    assert encoder is not None
    assert encoder.__class__.__name__ == "NormalizedClipEncoder"

    encoder = get_encoder("siglip", {})
    assert encoder is not None
    assert encoder.__class__.__name__ == "SiglipEncoder"

    encoder = get_encoder("dinov2siglip", {})
    assert encoder is not None
    assert encoder.__class__.__name__ == "Dinov2SigLIPEncoder"

    with pytest.raises(ValueError):
        get_encoder("invalid_encoder", {})


@pytest.mark.parametrize("encoder_name", encoders)
def test_get_encoder_all(encoder_name):
    print(f"Testing encoder: {encoder_name}")
    encoder = get_encoder(encoder_name, {})
    assert encoder is not None

    with pytest.raises(ValueError):
        get_encoder("invalid_encoder", {})

    for image_path in images:
        print(f"Testing encoder: {encoder_name} with image: {image_path}")
        encoder = get_encoder(encoder_name, {})
        assert encoder is not None

        image = Image.open(image_path)
        np_image = np.asarray(image)
        encoded = encoder.encode_image(np_image)
        assert encoded is not None
        assert isinstance(encoded, torch.Tensor)
        assert len(encoded) > 0
        print(f"Encoded: {encoded}")


if __name__ == "__main__":
    test_get_encoder()
    test_get_encoder_all("clip")
    test_get_encoder_all("normalized_clip")
    test_get_encoder_all("siglip")
    test_get_encoder_all("dinov2siglip")
