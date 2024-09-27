# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

from typing import Any, Dict, Optional, Union

from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

from stretch.llms.base import AbstractLLMClient, AbstractPromptBuilder


class MolmoClient(AbstractLLMClient):
    def __init__(
        self,
        prompt: Union[str, AbstractPromptBuilder],
        prompt_kwargs: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
    ):

        if model is None:
            model = "allenai/MolmoE-1B-0924"
        super().__init__(prompt, prompt_kwargs)
        # load the processor
        self.processor = AutoProcessor.from_pretrained(
            model, trust_remote_code=True, torch_dtype="auto", device_map="auto"
        )

        # load the model
        self.model = AutoModelForCausalLM.from_pretrained(
            model, trust_remote_code=True, torch_dtype="auto", device_map="auto"
        )

    def __call__(self, command: str, image: Optional[Image.Image] = None, verbose: bool = False):
        # process the image and text
        inputs = self.processor.process(images=[image] if image is not None else [], text=command)

        # move inputs to the correct device and make a batch of size 1
        inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}

        # generate output; maximum 200 new tokens; stop generation when <|endoftext|> is generated
        output = self.model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
            tokenizer=self.processor.tokenizer,
        )

        # only get generated tokens; decode them to text
        generated_tokens = output[0, inputs["input_ids"].size(1) :]
        generated_text = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return generated_text


if __name__ == "__main__":
    import requests

    image = Image.open(requests.get("https://picsum.photos/id/237/536/354", stream=True).raw)

    # import matplotlib.pyplot as plt

    client = MolmoClient(prompt=None)

    generated_text = client(command="Describe this image.", image=image)

    # print the generated text
    # >>> This photograph captures a small black puppy, likely a Labrador or a similar breed,
    #     sitting attentively on a weathered wooden deck. The deck, composed of three...
    print(generated_text)
