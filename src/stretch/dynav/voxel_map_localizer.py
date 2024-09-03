import numpy as np

import cv2

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

import clip

from stretch.dynav.mapping_utils import VoxelizedPointcloud

from typing import List, Optional, Tuple, Union
from torch import Tensor

from transformers import AutoProcessor, AutoModel

from sklearn.cluster import DBSCAN

# from ultralytics import YOLOWorld
from transformers import Owlv2Processor, Owlv2ForObjectDetection

import math
from PIL import Image

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

class VoxelMapLocalizer():
    def __init__(self, voxel_map_wrapper = None, exist_model = True, clip_model = None, processor = None, device = 'cuda', siglip = True):
        print('Localizer V3')
        self.voxel_map_wrapper = voxel_map_wrapper
        self.device = device
        # self.clip_model, self.preprocessor = clip.load(model_config, device=device)
        self.siglip = siglip
        if clip_model is not None and processor is not None:
            self.clip_model, self.preprocessor = clip_model, processor
        elif not self.siglip:
            self.clip_model, self.preprocessor = clip.load("ViT-B/16", device=self.device)
            self.clip_model.eval()
        else:
            self.clip_model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384").to(self.device)
            self.preprocessor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
            self.clip_model.eval()
        self.voxel_pcd = VoxelizedPointcloud(voxel_size = 0.05).to(self.device)
        # self.exist_model = YOLOWorld("yolov8l-worldv2.pt")
        self.existence_checking_model = exist_model
        if exist_model:
            print('WE ARE USING OWLV2!')
            self.exist_processor = AutoProcessor.from_pretrained("google/owlv2-base-patch16-ensemble")
            self.exist_model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(self.device)
        else:
            print('YOU ARE USING NOTHING!')
        

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

    def calculate_clip_and_st_embeddings_for_queries(self, queries):
        if isinstance(queries, str):
            queries = [queries] 
        if self.siglip:
            inputs = self.preprocessor(text=queries, padding="max_length", return_tensors="pt")
            for input in inputs:
                inputs[input] = inputs[input].to(self.clip_model.device)
            all_clip_tokens = self.clip_model.get_text_features(**inputs)
        else:
            text = clip.tokenize(queries).to(self.clip_model.device)
            all_clip_tokens = self.clip_model.encode_text(text)
        # text = clip.tokenize(queries).to(self.device)
        # all_clip_tokens = self.clip_model.encode_text(text)
        all_clip_tokens = F.normalize(all_clip_tokens, p=2, dim=-1)
        return all_clip_tokens
        
    def find_alignment_over_model(self, queries):
        clip_text_tokens = self.calculate_clip_and_st_embeddings_for_queries(queries)
        points, features, weights, _ = self.voxel_pcd.get_pointcloud()
        if points is None:
            return None
        features = F.normalize(features, p=2, dim=-1)
        point_alignments = clip_text_tokens.float() @ features.float().T
    
        # print(point_alignments.shape)
        return point_alignments

    def find_alignment_for_A(self, A):
        points, features, _, _ = self.voxel_pcd.get_pointcloud()
        alignments = self.find_alignment_over_model(A).cpu()
        return points[alignments.argmax(dim = -1)].detach().cpu()
    
    def find_obs_id_for_A(self, A):
        obs_counts = self.voxel_pcd._obs_counts
        alignments = self.find_alignment_over_model(A).cpu()
        return obs_counts[alignments.argmax(dim = -1)].detach().cpu()

    def compute_coord(self, text, obs_id, threshold = 0.2):
        if obs_id <= 0:
            return None
        rgb = self.voxel_map_wrapper.observations[obs_id - 1].rgb
        rgb = rgb.permute(2, 0, 1).to(torch.uint8)
        inputs = self.exist_processor(text=[['a photo of a ' + text]], images=rgb, return_tensors="pt")
        for input in inputs:
            inputs[input] = inputs[input].to('cuda')
        
        with torch.no_grad():
            outputs = self.exist_model(**inputs)
    
        target_sizes = torch.Tensor([rgb.size()[-2:]]).to(self.device)
        results = self.exist_processor.image_processor.post_process_object_detection(
            outputs, threshold=threshold, target_sizes=target_sizes
        )[0]
        depth = self.voxel_map_wrapper.observations[obs_id - 1].depth
        for idx, (score, bbox) in enumerate(sorted(zip(results['scores'], results['boxes']), key=lambda x: x[0], reverse=True)):
        
            tl_x, tl_y, br_x, br_y = bbox
            w, h = depth.shape
            # if w > h:
            #     tl_x, br_x = tl_x * w / h, br_x * w / h
            # else:
            #     tl_y, br_y = tl_y * h / w, br_y * h / w
            tl_x, tl_y, br_x, br_y = int(max(0, tl_x.item())), int(max(0, tl_y.item())), int(min(h, br_x.item())), int(min(w, br_y.item()))
            pose = self.voxel_map_wrapper.observations[obs_id - 1].camera_pose
            K = self.voxel_map_wrapper.observations[obs_id - 1].camera_K
            xyzs = get_xyz(depth, pose, K)[0]
            if torch.median(depth[tl_y: br_y, tl_x: br_x].reshape(-1)) < 3:
                return torch.median(xyzs[tl_y: br_y, tl_x: br_x].reshape(-1, 3), dim = 0).values
            # if depth[(tl_y + br_y) // 2, (tl_x + br_x) // 2] < 3.:
            #     return xyzs[(tl_y + br_y) // 2, (tl_x + br_x) // 2]
        return None

    def localize_A(self, A, debug = True, return_debug = False):
        points, _, _, _ = self.voxel_pcd.get_pointcloud()
        alignments = self.find_alignment_over_model(A).cpu()
        point = points[alignments.argmax(dim = -1)].detach().cpu().squeeze()
        obs_counts = self.voxel_pcd._obs_counts
        image_id = obs_counts[alignments.argmax(dim = -1)].detach().cpu()
        debug_text = ''
        target_point = None

        res = self.compute_coord(A, image_id)
        # res = None
        if res is not None:
            target_point = res
            debug_text += '#### - Obejct is detected in observations . **😃** Directly navigate to it.\n'
        else:
            # debug_text += '#### - Directly ignore this instance is the target. **😞** \n'
            if self.siglip:
                cosine_similarity_check = alignments.max().item() > 0.14
            else:
                cosine_similarity_check = alignments.max().item() > 0.3
            if cosine_similarity_check:
                target_point = point

                debug_text += '#### - The point has high cosine similarity. **😃** Directly navigate to it.\n'
            else:
                debug_text += '#### - Cannot verify whether this instance is the target. **😞** \n'
        # print('target_point', target_point)
        if not debug:
            return target_point
        elif not return_debug:
            return target_point, debug_text
        else:
            return target_point, debug_text, image_id, point