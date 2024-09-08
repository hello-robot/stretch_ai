from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import os

import torch
import numpy as np
from PIL import Image

from typing import Optional
from torch import Tensor

from stretch.utils.point_cloud_torch import unproject_masked_depth_to_xyz_coordinates
from stretch.dynav.mapping_utils import VoxelizedPointcloud

from transformers import AutoProcessor
from transformers import Owlv2ForObjectDetection

import google.generativeai as genai
from openai import OpenAI
import base64
from collections import OrderedDict

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
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
    fx, fy, ppx, ppy = intrinsics[..., 0, 0], intrinsics[..., 1, 1], intrinsics[..., 0, 2], intrinsics[..., 1, 2]
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

class LLM_Localizer():
    def __init__(self, voxel_map_wrapper = None, exist_model = 'gpt-4o', loc_model = 'owlv2', device = 'cuda'):
        self.voxel_map_wrapper = voxel_map_wrapper
        self.device = device
        self.voxel_pcd = VoxelizedPointcloud(voxel_size=0.2).to(self.device)
        # self.exist_model = YOLOWorld("yolov8l-worldv2.pt")
        self.existence_checking_model = exist_model
        if exist_model == 'gpt-4o':
            print('WE ARE USING OPENAI GPT4o')
            self.gpt_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        elif exist_model == 'gemini-1.5-pro':
            print('WE ARE USING GEMINI 1.5 PRO')
            
        elif exist_model == 'gemini-1.5-flash':
            print('WE ARE USING GEMINI 1.5 FLASH')
        else:
            print('YOU ARE USING NOTHING!')
        self.location_checking_model = loc_model
        if loc_model == 'owlv2':
            self.exist_processor = AutoProcessor.from_pretrained("google/owlv2-base-patch16-ensemble")
            self.exist_model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(self.device)
            print('WE ARE USING OWLV2 FOR LOCALIZATION!')
        else:
            print('YOU ARE USING VOXEL MAP FOR LOCALIZATION!')
        
    def add(self,
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
        self.voxel_pcd.add(points = points, 
                        features = features,
                        rgb = rgb,
                        weights = weights,
                        obs_count = obs_count)

    def compute_coord(self, text, image_info, threshold = 0.25):
        rgb = image_info['image']
        inputs = self.exist_processor(text=[['a photo of a ' + text]], images=rgb, return_tensors="pt")
        for input in inputs:
            inputs[input] = inputs[input].to('cuda')
        
        with torch.no_grad():
            outputs = self.exist_model(**inputs)
    
        target_sizes = torch.Tensor([rgb.size[::-1]]).to(self.device)
        results = self.exist_processor.image_processor.post_process_object_detection(
            outputs, threshold=threshold, target_sizes=target_sizes
        )[0]
        depth = image_info['depth']
        xyzs = image_info['xyz']
        for idx, (score, bbox) in enumerate(sorted(zip(results['scores'], results['boxes']), key=lambda x: x[0], reverse=True)):
        
            tl_x, tl_y, br_x, br_y = bbox
            w, h = depth.shape
            tl_x, tl_y, br_x, br_y = int(max(0, tl_x.item())), int(max(0, tl_y.item())), int(min(h, br_x.item())), int(min(w, br_y.item()))
            if np.median(depth[tl_y: br_y, tl_x: br_x].reshape(-1)) < 3:
                return torch.from_numpy(np.median(xyzs[tl_y: br_y, tl_x: br_x].reshape(-1, 3), axis = 0))
        return None

    def owl_locater(self, A, encoded_image, timestamps_lst):
        for i in timestamps_lst:
            image_info = encoded_image[i][-1]
            res = self.compute_coord(A, image_info, threshold=0.15)
            if res is not None:
                debug_text = '#### - Obejct is detected in observations where instance' + str(i + 1) + ' comes from. **😃** Directly navigate to it.\n'
                return res, debug_text, i, None

        debug_text = '#### - All instances are not the target! Maybe target object has not been observed yet. **😭**\n'
        return None, debug_text, None, None
    
    def gpt_chunk(self, chunk, sys_prompt, user_prompt):
        for i in range(10):
            try:
                response = self.gpt_client.chat.completions.create(
                    model=self.existence_checking_model,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt},
                        {"role": "user", "content": chunk}
                    ],
                    temperature=0.0,
                )                
                timestamps = response.choices[0].message.content
                if 'None' in timestamps:
                    return None
                else:
                    return list(map(int, timestamps.replace(' ', '').split(':')[1].split(',')))
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(10)
        return "Execution Failed"

    def gemini_chunk(self, chunk, sys_prompt, user_prompt):
        if self.existence_checking_model == 'gemini-1.5-pro':
            model_name="models/gemini-1.5-pro-exp-0827"
        elif self.existence_checking_model == 'gemini-1.5-flash':
            model_name="models/gemini-1.5-flash-exp-0827"

        for i in range(3):
            try:
                model = genai.GenerativeModel(model_name=model_name, system_instruction=sys_prompt)
                timestamps = model.generate_content(chunk + [user_prompt], generation_config=generation_config, safety_settings=safety_settings).text
                timestamps = timestamps.split('\n')[0]
                if 'None' in timestamps:
                    return None
                else:
                    return list(map(int, timestamps.replace(' ', '').split(':')[1].split(',')))
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(10)
        return "Execution Failed"



    def llm_locator(self, A, encoded_image, process_chunk, context_length = 30):
        timestamps_lst = []

        sys_prompt = f"""
        For each object query provided, list at most 10 timestamps that the object is most clearly shown. If the object does not appear, simply output the object name and the word "None" for the timestamp. Avoid any unnecessary explanations or additional formatting.
        
        Example:
        Input:
        The object you need to find are cat, car

        Output: 
        cat: 1,4,6,9
        car: None
        """

        user_prompt = f"""The object you need to find are {A}, car, bottle, shoes, watch, clothes, desk, chair, shoes, cup"""
        if 'gpt' in self.existence_checking_model:
            content = [item for sublist in list(encoded_image.values()) for item in sublist[:2]][-120:] # adjust to [-60:] for taking only the last 30 and have faster speed
        elif 'gemini' in self.existence_checking_model:
            content = [item for sublist in list(encoded_image.values()) for item in sublist[0]][-120:]
        content_chunks = [content[i:i + 2 * context_length] for i in range(0, len(content), 2 * context_length)]
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_to_chunk = {executor.submit(process_chunk, chunk, sys_prompt, user_prompt): chunk for chunk in content_chunks}
            
            for future in as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                try:
                    result = future.result()
                    if result:
                        timestamps_lst.extend(result)    
                except Exception as e:
                    print(f"Exception occurred: {e}")
        timestamps_lst = sorted(timestamps_lst, reverse=True)
        # print(A, timestamps_lst)
        return self.owl_locater(A, encoded_image, timestamps_lst)

    def localize_A(self, A, debug = True, return_debug = False, count_threshold = 3):
        encoded_image = OrderedDict()

        counts = torch.bincount(self.voxel_pcd._obs_counts)
        cur_obs = max(self.voxel_pcd._obs_counts)
        filtered_obs = (counts > count_threshold).nonzero(as_tuple=True)[0].tolist()
        # filtered_obs = sorted(set(filtered_obs + [i for i in range(cur_obs-10, cur_obs+1)]))
        filtered_obs = sorted(filtered_obs)

        # filtered_obs = (counts <= count_threshold).nonzero(as_tuple=True)[0].tolist()
        # filtered_obs = [obs for obs in filtered_obs if (cur_obs - obs) >= 10]

        if 'gemini' in self.existence_checking_model:
            process_chunk = self.gemini_chunk
            context_length = 100
        elif 'gpt' in self.existence_checking_model:
            process_chunk = self.gpt_chunk
            context_length = 30

        for obs_id in filtered_obs: 
            obs_id -= 1
            rgb = self.voxel_map_wrapper.observations[obs_id].rgb.numpy()
            depth = self.voxel_map_wrapper.observations[obs_id].depth
            camera_pose = self.voxel_map_wrapper.observations[obs_id].camera_pose
            camera_K = self.voxel_map_wrapper.observations[obs_id].camera_K

            # full_world_xyz = unproject_masked_depth_to_xyz_coordinates(  # Batchable!
            #     depth=depth.unsqueeze(0).unsqueeze(1),
            #     pose=camera_pose.unsqueeze(0),
            #     inv_intrinsics=torch.linalg.inv(camera_K[:3, :3]).unsqueeze(0),
            # )
            xyz = get_xyz(depth, camera_pose, camera_K)[0]
            # print(full_world_xyz.shape)
            depth = depth.numpy()
            # rgb[depth > 2.5] = [0, 0, 0]
            image = Image.fromarray(rgb.astype(np.uint8), mode='RGB')
            if 'gemini' in self.existence_checking_model:
                encoded_image[obs_id] = [[f"Following is the image took on timestep {obs_id}: ", image], {'image':image, 'xyz': xyz, 'depth':depth}]
            elif 'gpt' in self.existence_checking_model:
                buffered = BytesIO()
                image.save(buffered, format="PNG")
                img_bytes = buffered.getvalue()
                base64_encoded = base64.b64encode(img_bytes).decode('utf-8')
                encoded_image[obs_id] = [{"type": "text", "text": f"Following is the image took on timestep {obs_id}"},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{base64_encoded}"}
                    }, {'image':image, 'xyz':xyz, 'depth':depth}]
        target_point, debug_text, obs, point = self.llm_locator(A, encoded_image, process_chunk, context_length)
        if not debug:
            return target_point
        elif not return_debug:
            return target_point, debug_text
        else:
            return target_point, debug_text, obs, point