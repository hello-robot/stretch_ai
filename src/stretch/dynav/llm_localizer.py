# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import base64
import os
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from typing import Optional

import google.generativeai as genai
import numpy as np
import torch
from openai import OpenAI
from PIL import Image
from torch import Tensor
from transformers import AutoProcessor, Owlv2ForObjectDetection

from stretch.dynav.mapping_utils.voxelized_pcd import VoxelizedPointcloud

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
generation_config = genai.GenerationConfig(temperature=0)
safety_settings = [
    {
        "category": "HARM_CATEGORY_DANGEROUS",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]


def get_inv_intrinsics(intrinsics):
    # return intrinsics.double().inverse().to(intrinsics)
    fx, fy, ppx, ppy = (
        intrinsics[..., 0, 0],
        intrinsics[..., 1, 1],
        intrinsics[..., 0, 2],
        intrinsics[..., 1, 2],
    )
    inv_intrinsics = torch.zeros_like(intrinsics)
    inv_intrinsics[..., 0, 0] = 1.0 / fx
    inv_intrinsics[..., 1, 1] = 1.0 / fy
    inv_intrinsics[..., 0, 2] = -ppx / fx
    inv_intrinsics[..., 1, 2] = -ppy / fy
    inv_intrinsics[..., 2, 2] = 1.0
    return inv_intrinsics


def get_xyz(depth, pose, intrinsics):
    """Returns the XYZ coordinates for a set of points.

    Args:
        depth: The depth array, with shape (B, 1, H, W)
        pose: The pose array, with shape (B, 4, 4)
        intrinsics: The intrinsics array, with shape (B, 3, 3)

    Returns:
        The XYZ coordinates of the projected points, with shape (B, H, W, 3)
    """
    if not isinstance(depth, torch.Tensor):
        depth = torch.from_numpy(depth)
    if not isinstance(pose, torch.Tensor):
        pose = torch.from_numpy(pose)
    if not isinstance(intrinsics, torch.Tensor):
        intrinsics = torch.from_numpy(intrinsics)
    while depth.ndim < 4:
        depth = depth.unsqueeze(0)
    while pose.ndim < 3:
        pose = pose.unsqueeze(0)
    while intrinsics.ndim < 3:
        intrinsics = intrinsics.unsqueeze(0)
    (bsz, _, height, width), device, dtype = depth.shape, depth.device, intrinsics.dtype

    # Gets the pixel grid.
    xs, ys = torch.meshgrid(
        torch.arange(0, width, device=device, dtype=dtype),
        torch.arange(0, height, device=device, dtype=dtype),
        indexing="xy",
    )
    xy = torch.stack([xs, ys], dim=-1).flatten(0, 1).unsqueeze(0).repeat_interleave(bsz, 0)
    xyz = torch.cat((xy, torch.ones_like(xy[..., :1])), dim=-1)

    # Applies intrinsics and extrinsics.
    # xyz = xyz @ intrinsics.inverse().transpose(-1, -2)
    xyz = xyz @ get_inv_intrinsics(intrinsics).transpose(-1, -2)
    xyz = xyz * depth.flatten(1).unsqueeze(-1)
    xyz = (xyz[..., None, :] * pose[..., None, :3, :3]).sum(dim=-1) + pose[..., None, :3, 3]

    xyz = xyz.unflatten(1, (height, width))

    return xyz


class LLM_Localizer:
    def __init__(
        self, voxel_map_wrapper=None, exist_model="gemini-1.5-pro", loc_model="owlv2", device="cuda"
    ):
        self.voxel_map_wrapper = voxel_map_wrapper
        self.device = device
        self.voxel_pcd = VoxelizedPointcloud(voxel_size=0.2).to(self.device)
        self.existence_checking_model = exist_model

        self.api_key_1 = os.getenv("GOOGLE_API_KEY")
        self.api_key_2 = os.getenv("GOOGLE_API_KEY_2")
        self.api_key_3 = os.getenv("GOOGLE_API_KEY_3")
        self.api_key = self.api_key_1

        self.context_length = 60
        self.count_threshold = 3
        if "gpt" in self.existence_checking_model:
            self.max_img_per_request = 30
        else:
            self.max_img_per_request = 200

        if exist_model == "gpt-4o":
            print("WE ARE USING OPENAI GPT4o")
            self.gpt_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif exist_model == "gemini-1.5-pro":
            print("WE ARE USING GEMINI 1.5 PRO")

        elif exist_model == "gemini-1.5-flash":
            print("WE ARE USING GEMINI 1.5 FLASH")
        else:
            print("YOU ARE USING NOTHING!")
        self.location_checking_model = loc_model
        if loc_model == "owlv2":
            self.exist_processor = AutoProcessor.from_pretrained(
                "google/owlv2-base-patch16-ensemble"
            )
            self.exist_model = Owlv2ForObjectDetection.from_pretrained(
                "google/owlv2-base-patch16-ensemble"
            ).to(self.device)
            print("WE ARE USING OWLV2 FOR LOCALIZATION!")
        else:
            print("YOU ARE USING VOXEL MAP FOR LOCALIZATION!")

    def add(
        self,
        points: Tensor,
        features: Optional[Tensor],
        rgb: Optional[Tensor],
        weights: Optional[Tensor] = None,
        obs_count: Optional[Tensor] = None,
    ):
        points = points.to(self.device)
        if features is not None:
            features = features.to(self.device)
        if rgb is not None:
            rgb = rgb.to(self.device)
        if weights is not None:
            weights = weights.to(self.device)
        self.voxel_pcd.add(
            points=points, features=features, rgb=rgb, weights=weights, obs_count=obs_count
        )

    def compute_coord(self, text, image_info, threshold=0.2):
        rgb = image_info["image"]
        inputs = self.exist_processor(text=text, images=rgb, return_tensors="pt")
        for input in inputs:
            inputs[input] = inputs[input].to("cuda")

        with torch.no_grad():
            outputs = self.exist_model(**inputs)

        target_sizes = torch.Tensor([rgb.size[::-1]]).to(self.device)
        results = self.exist_processor.image_processor.post_process_object_detection(
            outputs, threshold=threshold, target_sizes=target_sizes
        )[0]
        depth = image_info["depth"]
        xyzs = image_info["xyz"]
        temp_lst = []
        for idx, (score, bbox) in enumerate(
            sorted(zip(results["scores"], results["boxes"]), key=lambda x: x[0], reverse=True)
        ):

            tl_x, tl_y, br_x, br_y = bbox
            w, h = depth.shape
            tl_x, tl_y, br_x, br_y = (
                int(max(0, tl_x.item())),
                int(max(0, tl_y.item())),
                int(min(h, br_x.item())),
                int(min(w, br_y.item())),
            )
            if np.median(depth[tl_y:br_y, tl_x:br_x].reshape(-1)) < 3:
                coordinate = torch.from_numpy(
                    np.median(xyzs[tl_y:br_y, tl_x:br_x].reshape(-1, 3), axis=0)
                )
                # temp_lst.append(coordinate)
                return coordinate
        return None
        # return temp_lst

    def owl_locater(self, A, encoded_image, timestamps_lst):
        # query_coord_lst = []
        # # dbscan = DBSCAN(eps=1.5, min_samples=1)
        # centroid = None
        for i in timestamps_lst:
            if i in encoded_image:
                image_info = encoded_image[i][-1]
                res = self.compute_coord(A, image_info, threshold=0.2)
                if res is not None:
                    debug_text = (
                        "#### - Object is detected in observations where instance"
                        + str(i + 1)
                        + " comes from. **😃** Directly navigate to it.\n"
                    )
                    return res, debug_text, i, None
        debug_text = "#### - All instances are not the target! Maybe target object has not been observed yet. **😭**\n"

        # if query_coord_lst != []:
        #     query_coord_lst = np.array(query_coord_lst)
        #     dbscan.fit(query_coord_lst)
        #     labels = dbscan.labels_
        #     unique_labels = set(labels) - {-1}
        #     largest_cluster_label = None
        #     largest_cluster_size = 0
        #     for label in unique_labels:
        #         cluster_size = np.sum(labels == label)
        #         if cluster_size > largest_cluster_size:
        #             largest_cluster_size = cluster_size
        #             largest_cluster_label = label
        #     largest_cluster_points = query_coord_lst[labels == largest_cluster_label]
        #     centroid = largest_cluster_points.mean(axis=0)
        #     return centroid, debug_text, None, None
        return None, debug_text, None, None

    def process_chunk(self, chunk, sys_prompt, user_prompt):
        for i in range(50):
            try:
                if "gpt" in self.existence_checking_model:
                    start_time = time.time()
                    response = (
                        self.gpt_client.chat.completions.create(
                            model=self.existence_checking_model,
                            messages=[
                                {"role": "system", "content": sys_prompt},
                                {"role": "user", "content": user_prompt},
                                {"role": "user", "content": chunk},
                            ],
                            temperature=0.0,
                        )
                        .choices[0]
                        .message.content
                    )

                    end_time = time.time()
                    print("GPT request cost:", end_time - start_time)
                else:
                    model = genai.GenerativeModel(
                        model_name=f"models/{self.existence_checking_model}-exp-0827",
                        system_instruction=sys_prompt,
                    )
                    # "models/{self.existence_checking_model}-exp-0827"
                    response = model.generate_content(
                        chunk + [user_prompt],
                        generation_config=generation_config,
                        safety_settings=safety_settings,
                    ).text
                # print("Assistant: ", response)
                return response
            except Exception as e:
                if self.api_key == self.api_key_1:
                    self.api_key = self.api_key_2
                elif self.api_key == self.api_key_2:
                    self.api_key = self.api_key_3
                elif self.api_key == self.api_key_3:
                    self.api_key = self.api_key_1
                genai.configure(api_key=self.api_key)
                print(f"Error: {e}")
                # time.sleep(10)
        print("Execution Failed")
        return "Execution Failed"

    def update_dict(self, A, timestamps_dict, content):
        timestamps_lst = content.split("\n")
        for item in timestamps_lst:
            if len(item) < 3 or ":" not in item:
                continue
            key, value_str = item.split(":")
            if A not in key:
                continue
            if "None" in value_str:
                value = None
            else:
                value = list(map(int, value_str.replace(" ", "").split(",")))
            if key in timestamps_dict:
                if timestamps_dict[key] is None:
                    timestamps_dict[key] = value
                elif value is not None:
                    timestamps_dict[key].extend(value)
            else:
                timestamps_dict[key] = value

    def llm_locator(self, A, encoded_image):
        timestamps_dict = {}

        sys_prompt = f"""
        For object query I give, you need to find timestamps of images that the object is shown, without any unnecessary explanation or space. If the object never exist, please output the object name and the word "None" for timestamps.

        Example:
        Input:
        The object you need to find is blue bottle, orange chair, white box

        Output: 
        blue bottle: 3,6,9
        orange chair: None
        white box: 2,4,10"""

        user_prompt = f"""The object you need to find is {A}"""
        if "gpt" in self.existence_checking_model:
            content = [item for sublist in list(encoded_image.values()) for item in sublist[:2]][
                -self.context_length * 2 :
            ]
        else:
            content = [item for sublist in list(encoded_image.values()) for item in sublist[0]][
                -self.context_length * 2 :
            ]

        content_chunks = [
            content[i : i + 2 * self.max_img_per_request]
            for i in range(0, len(content), 2 * self.max_img_per_request)
        ]

        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_chunk = {
                executor.submit(self.process_chunk, chunk, sys_prompt, user_prompt): chunk
                for chunk in content_chunks
            }

            for future in as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                try:
                    result = future.result()
                    if result:
                        self.update_dict(A, timestamps_dict, result)
                except Exception as e:
                    print(f"Exception occurred: {e}")
        if A not in timestamps_dict:
            return None, "debug_text", None, None

        timestamps_lst = timestamps_dict[A]
        if timestamps_lst is None:
            return None, "debug_text", None, None
        timestamps_lst = sorted(timestamps_lst, reverse=True)
        # return None
        return self.owl_locater(A, encoded_image, timestamps_lst)

    def localize_A(self, A, debug=True, return_debug=False):
        encoded_image = OrderedDict()
        counts = torch.bincount(self.voxel_pcd._obs_counts)
        filtered_obs = (counts > self.count_threshold).nonzero(as_tuple=True)[0].tolist()
        filtered_obs = sorted(filtered_obs)

        for obs_id in filtered_obs:
            obs_id -= 1
            rgb = np.copy(self.voxel_map_wrapper.observations[obs_id].rgb.numpy())
            depth = self.voxel_map_wrapper.observations[obs_id].depth
            camera_pose = self.voxel_map_wrapper.observations[obs_id].camera_pose
            camera_K = self.voxel_map_wrapper.observations[obs_id].camera_K
            xyz = get_xyz(depth, camera_pose, camera_K)[0]
            depth = depth.numpy()

            rgb[depth > 2.5] = [0, 0, 0]

            image = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
            if "gemini" in self.existence_checking_model:
                encoded_image[obs_id] = [
                    [f"Following is the image took on timestamp {obs_id}: ", image],
                    {"image": image, "xyz": xyz, "depth": depth},
                ]
            elif "gpt" in self.existence_checking_model:
                buffered = BytesIO()
                image.save(buffered, format="PNG")
                img_bytes = buffered.getvalue()
                base64_encoded = base64.b64encode(img_bytes).decode("utf-8")
                encoded_image[obs_id] = [
                    {
                        "type": "text",
                        "text": f"Following is the image took on timestamp {obs_id}: ",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_encoded}"},
                    },
                    {"image": image, "xyz": xyz, "depth": depth},
                ]
                # print(obs_id)

        start_time = time.time()
        target_point, debug_text, obs, point = self.llm_locator(A, encoded_image)
        end_time = time.time()
        print("It takes", end_time - start_time)

        if not debug:
            return target_point
        elif not return_debug:
            return target_point, debug_text
        else:
            return target_point, debug_text, obs, point
