# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import base64
import io
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image
from torchvision import transforms

logging.basicConfig(level=logging.INFO)


class MultiCropOpenAIClient:
    # TODO: inherit from AbstractLLMClient
    def __init__(self, cfg):
        self.cfg = cfg
        self.prompt = self.cfg["prompt"]
        self.api_key = self.cfg["api_key"]
        self.max_tokens = self.cfg["max_tokens"]
        self.temperature = self.cfg["temperature"]
        self.to_pil = transforms.ToPILImage()
        self.resize = transforms.Resize((self.cfg["img_size"], self.cfg["img_size"]))

    def reset(self):
        self.errors = {}
        self.responses = {}
        self.current_round = 0
        self.goal = None

    def _prepare_samples(self, obs, goal, debug_path=None):
        context_messages = [{"type": "text", "text": self.prompt}]
        self.goal = goal
        for img_id, object_image in enumerate(obs.object_images):
            # Convert to base64Image
            idx = object_image.crop_id
            pil_image = self.resize(Image.fromarray(np.array(object_image.image, dtype=np.uint8)))

            plt.subplot(1, len(obs.object_images), img_id + 1)
            plt.imshow(pil_image)
            plt.axis("off")
            image_bytes = io.BytesIO()
            if debug_path:
                round_path = os.path.join(debug_path, str(self.current_round))
                os.makedirs(round_path, exist_ok=True)
                pil_image.save(os.path.join(round_path, str(img_id) + ".png"))
            pil_image.save(image_bytes, format="png")
            base64_image = base64.b64encode(image_bytes.getvalue()).decode("utf-8")

            # Write context images
            text_pre = {"type": "text", "text": f"<img_{idx}>"}
            if idx == len(obs.object_images) - 1:
                text_post = {"type": "text", "text": f"</img_{idx}>"}
            else:
                text_post = {"type": "text", "text": f"</img_{idx}>, "}
            text_img = {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_image}"},
            }
            context_messages.append(text_pre)
            context_messages.append(text_img)
            context_messages.append(text_post)

        scene_graph_text = "\n2. Scene descriptions: "
        if obs.scene_graph:
            for rel in obs.scene_graph:
                crop_id_a = next(
                    (
                        obj_crop.crop_id
                        for obj_crop in obs.object_images
                        if obj_crop.instance_id == rel[0]
                    ),
                    None,
                )
                crop_id_b = next(
                    (
                        obj_crop.crop_id
                        for obj_crop in obs.object_images
                        if obj_crop.instance_id == rel[1]
                    ),
                    None,
                )
                scene_graph_text += f"img_{crop_id_a} is {rel[2]} img_{crop_id_b}; "
        context_messages.append(
            {
                "type": "text",
                "text": scene_graph_text + "\n",
            }
        )

        plt.suptitle(f"Prompts that are automatically generated for task: {self.goal}")
        plt.show()

        context_messages.append({"type": "text", "text": f"3. Query: {self.goal}\n"})
        context_messages.append({"type": "text", "text": "4. Answer: "})
        chat_input = {
            "model": "gpt-4-turbo",
            "messages": [{"role": "user", "content": context_messages}],
            "max_tokens": self.max_tokens,
        }
        return chat_input

    def _request(self, chat_input):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=chat_input,
        )
        json_res = response.json()
        print(f">>>>>> the original output from gpt is: {json_res} >>>>>>>>>")
        if "choices" in json_res:
            res = json_res["choices"][0]["message"]["content"]
        elif "error" in json_res:
            self.errors[self.current_round] = json_res
            return "gpt API error"
        # the prompt come with "Answer: " prefix
        self.responses[self.current_round] = res
        return res

    def act_on_observations(
        self,
        obs,
        goal=None,
        debug_path=None,
    ):
        if not obs:
            raise RuntimeError("no object-centric visual observations!")
        self.current_round += 1
        chat_input = self._prepare_samples(obs, goal, debug_path=debug_path)
        return self._request(chat_input)
