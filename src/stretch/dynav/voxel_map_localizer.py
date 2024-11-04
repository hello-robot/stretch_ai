# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import base64
import re
from io import BytesIO
from typing import Optional

import clip
import numpy as np
import torch
import torch.nn.functional as F
from openai import OpenAI
from PIL import Image
from sklearn.cluster import DBSCAN
from torch import Tensor
from transformers import AutoModel, AutoProcessor, Owlv2ForObjectDetection

from stretch.utils.logger import Logger
from stretch.utils.voxel import VoxelizedPointcloud

# from ultalytics import YOLOWorld


# Create a logger
logger = Logger(__name__)


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


class VoxelMapLocalizer:
    """This localizes a query in the voxel map."""

    model_choices = ["gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini"]

    def __init__(
        self,
        voxel_map_wrapper=None,
        exist_model=True,
        clip_model=None,
        processor=None,
        device="cuda",
        siglip=True,
        gpt_model: str = "gpt-4o",
    ):
        logger.info("Localizer V3")
        self.voxel_map_wrapper = voxel_map_wrapper
        self.device = device
        # self.clip_model, self.preprocessor = clip.load(model_config, device=device)
        self.siglip = siglip

        self.gpt_model = gpt_model
        if self.gpt_model not in self.model_choices:
            raise ValueError(f"model must be one of {self.model_choices}, got {self.gpt_model}")

        if clip_model is not None and processor is not None:
            self.clip_model, self.preprocessor = clip_model, processor
        elif not self.siglip:
            self.clip_model, self.preprocessor = clip.load("ViT-B/16", device=self.device)
            self.clip_model.eval()
        else:
            self.clip_model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384").to(
                self.device
            )
            self.preprocessor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
            self.clip_model.eval()
        self.voxel_pcd = VoxelizedPointcloud(voxel_size=0.05).to(self.device)

        # OpenAI will automatically use the API key from the environment
        self.gpt_client = OpenAI()

        # self.yolo_model = YOLOWorld("yolov8s-worldv2.pt")

        self.exist_processor = AutoProcessor.from_pretrained("google/owlv2-base-patch16-ensemble")
        self.exist_model = Owlv2ForObjectDetection.from_pretrained(
            "google/owlv2-base-patch16-ensemble"
        ).to(self.device)

        # self.existence_checking_model = exist_model
        # if exist_model:
        #     logger.info("WE ARE USING OWLV2!")
        #     self.exist_processor = AutoProcessor.from_pretrained(
        #         "google/owlv2-large-patch14-ensemble"
        #     )
        #     self.exist_model = Owlv2ForObjectDetection.from_pretrained(
        #         "google/owlv2-large-patch14-ensemble"
        #     ).to(self.device)
        # else:
        #     logger.info("YOU ARE USING NOTHING!")

    def add(
        self,
        points: Tensor,
        features: Optional[Tensor],
        rgb: Optional[Tensor],
        weights: Optional[Tensor] = None,
        obs_count: Optional[Tensor] = None,
    ):
        """Adds a pointcloud to the voxel map.

        Args:
            points: The points tensor, with shape (N, 3)
            features: The features tensor, with shape (N, C)
            rgb: The RGB tensor, with shape (N, 3)
            weights: The weights tensor, with shape (N)
            obs_count: The observation count tensor, with shape (N)
        """
        points = points.to(self.device)
        if features is not None:
            features = features.to(self.device)
        if rgb is not None:
            rgb = rgb.to(self.device)
        if weights is not None:
            weights = weights.to(self.device)
        # if weight_decay is not None and self.voxel_pcd._weights is not None:
        #     self.voxel_pcd._weights *= weight_decay
        self.voxel_pcd.add(
            points=points, features=features, rgb=rgb, weights=weights, obs_count=obs_count
        )

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

        # text = self.tokenizer(queries, return_tensors="pt", padding=True).input_ids.to(self.device)
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
        return points[alignments.argmax(dim=-1)].detach().cpu()

    def find_obs_id_for_A(self, A):
        obs_counts = self.voxel_pcd._obs_counts
        alignments = self.find_alignment_over_model(A).cpu()
        return obs_counts[alignments.argmax(dim=-1)].detach().cpu()

    def compute_coord(self, text, obs_id, threshold=0.2):
        if obs_id <= 0 or obs_id > len(self.voxel_map_wrapper.observations):
            return None
        rgb = self.voxel_map_wrapper.observations[obs_id - 1].rgb.clone()
        pose = self.voxel_map_wrapper.observations[obs_id - 1].camera_pose
        depth = self.voxel_map_wrapper.observations[obs_id - 1].depth
        K = self.voxel_map_wrapper.observations[obs_id - 1].camera_K
        xyzs = get_xyz(depth, pose, K)[0]
        rgb[depth > 2.5] = 0

        rgb = rgb.permute(2, 0, 1).to(torch.uint8)
        inputs = self.exist_processor(
            text=[["a photo of a " + text]], images=rgb, return_tensors="pt"
        )
        for input in inputs:
            inputs[input] = inputs[input].to("cuda")

        with torch.no_grad():
            outputs = self.exist_model(**inputs)

        target_sizes = torch.Tensor([rgb.size()[-2:]]).to(self.device)
        results = self.exist_processor.image_processor.post_process_object_detection(
            outputs, threshold=threshold, target_sizes=target_sizes
        )[0]

        # self.yolo_model.set_classes([text])
        # print(threshold)
        # results = self.yolo_model.predict(rgb.numpy(), conf=threshold, verbose=False)

        xyxy = results["boxes"]
        scores = results["scores"]

        # xyxy = results[0].boxes.xyxy
        # scores = results[0].boxes.conf
        if len(xyxy) > 0:
            bbox = xyxy[torch.argmax(scores)]
            tl_x, tl_y, br_x, br_y = bbox
            w, h = depth.shape
            tl_x, tl_y, br_x, br_y = (
                int(max(0, tl_x.item())),
                int(max(0, tl_y.item())),
                int(min(h, br_x.item())),
                int(min(w, br_y.item())),
            )
            return torch.median(xyzs[tl_y:br_y, tl_x:br_x].reshape(-1, 3), dim=0).values
        else:
            return None
        # for idx, (score, bbox) in enumerate(
        #     sorted(zip(results["scores"], results["boxes"]), key=lambda x: x[0], reverse=True)
        # ):

        #     tl_x, tl_y, br_x, br_y = bbox
        #     w, h = depth.shape
        #     tl_x, tl_y, br_x, br_y = (
        #         int(max(0, tl_x.item())),
        #         int(max(0, tl_y.item())),
        #         int(min(h, br_x.item())),
        #         int(min(w, br_y.item())),
        #     )

        #     if torch.min(depth[tl_y:br_y, tl_x:br_x].reshape(-1)) < 2.5:
        #         return torch.median(xyzs[tl_y:br_y, tl_x:br_x].reshape(-1, 3), dim=0).values
        # return None

    def verify_point(self, A, point, distance_threshold=0.2, similarity_threshold=0.13):
        if isinstance(point, np.ndarray):
            point = torch.from_numpy(point)
        points, _, _, _ = self.voxel_pcd.get_pointcloud()
        distances = torch.linalg.norm(point - points.detach().cpu(), dim=-1)
        if torch.min(distances) > distance_threshold:
            print("Points are so far from other points!")
            return False
        alignments = self.find_alignment_over_model(A).detach().cpu()[0]
        if torch.max(alignments[distances <= distance_threshold]) < similarity_threshold:
            print("Points close the the point are not similar to the text!")
        return torch.max(alignments[distances < distance_threshold]) >= similarity_threshold

    def find_all_images(self, A):
        points, _, _, _ = self.voxel_pcd.get_pointcloud()
        points = points.cpu()
        alignments = self.find_alignment_over_model(A).cpu().squeeze()
        obs_counts = self.voxel_pcd._obs_counts.cpu()

        turning_point = min(0.12, alignments[torch.argsort(alignments)[-100]])
        mask = alignments >= turning_point
        obs_counts = obs_counts[mask]
        alignments = alignments[mask]
        points = points[mask]

        unique_obs_counts, inverse_indices = torch.unique(obs_counts, return_inverse=True)

        points_with_max_alignment = torch.zeros((len(unique_obs_counts), points.size(1)))
        max_alignments = torch.zeros(len(unique_obs_counts))

        for i in range(len(unique_obs_counts)):
            # Get indices of elements belonging to the current cluster
            indices_in_cluster = (inverse_indices == i).nonzero(as_tuple=True)[0]
            if len(indices_in_cluster) <= 2:
                continue

            # Extract the alignments and points for the current cluster
            cluster_alignments = alignments[indices_in_cluster].squeeze()
            cluster_points = points[indices_in_cluster]

            # Step 3: Find the point with the highest alignment in the cluster
            max_alignment_idx_in_cluster = cluster_alignments.argmax()
            point_with_max_alignment = cluster_points[max_alignment_idx_in_cluster]
            # point_with_max_alignment = torch.median(cluster_points, dim = 0).values
            # dbscan = DBSCAN(eps=0.25, min_samples=5)
            # clusters = dbscan.fit(cluster_points)
            # labels = torch.Tensor(clusters.labels_)
            # cluster_ids, counts = torch.unique(labels, return_counts = True)
            # cluster_id = cluster_ids[torch.argmax(counts)]
            # point_with_max_alignment = torch.mean(cluster_points[labels == cluster_id], axis = 0)

            # Store the result
            points_with_max_alignment[i] = point_with_max_alignment
            max_alignments[i] = cluster_alignments.max()

        top_alignments, top_indices = torch.topk(
            max_alignments, k=min(3, len(max_alignments)), dim=0, largest=True, sorted=True
        )
        top_points = points_with_max_alignment[top_indices]
        top_obs_counts = unique_obs_counts[top_indices]

        sorted_obs_counts, sorted_indices = torch.sort(top_obs_counts, descending=False)
        sorted_points = top_points[sorted_indices]
        top_alignments = top_alignments[sorted_indices]

        return sorted_obs_counts, sorted_points, top_alignments

    def llm_locator(self, image_ids, A):
        sys_prompt = f"""
        For object query I give, you need to find images that the object is shown. You should first caption each image and then make conclusion.

        Example #1:
            Input:
                The object you need to find is blue bottle.
            Output:
                Caption:
                    Image 1 is a red bottle. Image 2 is a blue mug. Image 3 is a blue bag. Image 4 is a blue bottle. Image 5 is a blue bottle
                Images: 
                    4, 5

        Example #2:
            Input:
                The object you need to find is orange cup.
            Output:
                Caption:
                    Image 1 is a orange fruit. Image 2 is a orange sofa. Image 3 is a blue cup.
                Images:
                    None

        Example #3:
            Input:
                The object you need to find is potato chip
            Output:
                Caption:
                    Image 1 is a sofa. Image 2 is a potato chip. Image 3 is a pepper. Image 4 is a laptop.
                Images:
                1"""
        user_prompt = f"""The object you need to find is {A}"""

        system_messages = [{"type": "text", "text": sys_prompt}]
        user_messages = []
        for obs_id in image_ids:
            obs_id = int(obs_id) - 1
            rgb = np.copy(self.voxel_map_wrapper.observations[obs_id].rgb.numpy())
            depth = self.voxel_map_wrapper.observations[obs_id].depth
            rgb[depth > 2.5] = [0, 0, 0]
            image = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
            buffered = BytesIO()
            # image.show()
            image.save(buffered, format="PNG")
            img_bytes = buffered.getvalue()
            base64_encoded = base64.b64encode(img_bytes).decode("utf-8")
            user_messages.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_encoded}",
                        "detail": "high",
                    },
                }
            )
        user_messages.append(
            {
                "type": "text",
                "text": "The object you need to find is " + A,
            }
        )

        response = (
            self.gpt_client.chat.completions.create(
                model=self.gpt_model,
                messages=[
                    {"role": "system", "content": system_messages},
                    {"role": "user", "content": user_messages},
                ],
                temperature=0.0,
            )
            .choices[0]
            .message.content
        )
        return self.process_response(response)

    def process_response(self, response):
        try:
            # Use regex to locate the 'Images:' section, allowing for varying whitespace and line breaks
            images_section_match = re.search(r"Images:\s*([\s\S]+)", response, re.IGNORECASE)
            if not images_section_match:
                raise ValueError("The 'Images:' section is missing.")

            # Extract the content after 'Images:'
            images_content = images_section_match.group(1).strip()

            # Check if the content is 'None' (case-insensitive)
            if images_content.lower() == "none":
                return None

            # Use regex to find all numbers, regardless of separators like commas, periods, or spaces
            numbers = re.findall(r"\d+", images_content)

            if not numbers:
                raise ValueError("No numbers found in the 'Images:' section.")

            # Convert all found numbers to integers
            numbers = [int(num) for num in numbers]

            # Return all numbers as a list if multiple numbers are found
            if len(numbers) > 0:
                return numbers[-1]
            else:
                return None

        except Exception as e:
            # Handle any exceptions and optionally log the error message
            print(f"Error: {e}")
            return None

    def localize_A(self, A, debug=True, return_debug=False):
        points, _, _, _ = self.voxel_pcd.get_pointcloud()
        alignments = self.find_alignment_over_model(A).cpu()
        # alignments = alignments[0, points[:, -1] >= 0.1]
        # points = points[points[:, -1] >= 0.1]
        point = points[alignments.argmax(dim=-1)].detach().cpu().squeeze()
        obs_counts = self.voxel_pcd._obs_counts
        image_id = obs_counts[alignments.argmax(dim=-1)].detach().cpu()
        debug_text = ""
        target_point = None

        image_ids, points, alignments = self.find_all_images(A)
        # Check cosine similarity first, it is not reasonable to always use gpt
        # if torch.max(alignments) < 0.08:
        #     # print('ALIGNMENTS TOO SMALL!\n')
        #     target_id = None
        # # elif torch.max(alignments) > 0.16:
        # #     print('ALIGNMENTS TOO LARGE!\n')
        # #     target_id = torch.argmax(alignments)
        # else:
        #     target_id = self.llm_locator(image_ids, A)
        target_id = self.llm_locator(image_ids, A)

        if target_id is None:
            debug_text += "#### - Cannot verify whether this instance is the target. **😞** \n"
            image_id = None
            point = None
        else:
            target_id -= 1
            target_point = points[target_id]
            image_id = image_ids[target_id]
            point = points[target_id]
            debug_text += "#### - An image is identified \n"

        if image_id is not None:
            res = self.compute_coord(A, image_id, threshold=0.01)
            if res is not None:
                target_point = res
            else:
                target_point = point

        if not debug:
            return target_point
        elif not return_debug:
            return target_point, debug_text
        else:
            return target_point, debug_text, image_id, point

    def find_clusters_for_A(self, A, return_obs_counts=False, debug=False, turning_point=None):

        debug_text = ""

        points, features, _, _ = self.voxel_pcd.get_pointcloud()
        alignments = self.find_alignment_over_model(A).cpu().reshape(-1).detach().numpy()
        if turning_point is None:
            if self.siglip:
                turning_point = min(0.14, alignments[np.argsort(alignments)[-20]])
            else:
                turning_point = min(0.3, alignments[np.argsort(alignments)[-20]])
        mask = alignments >= turning_point
        alignments = alignments[mask]
        points = points[mask]
        if len(points) == 0:

            debug_text += (
                "### - No instance found! Maybe target object has not been observed yet. **😭**\n"
            )

            output = [[], [], [], []]
            if return_obs_counts:
                output.append([])
            if debug:
                output.append(debug_text)

            return output
        else:
            if return_obs_counts:
                obs_ids = self.voxel_pcd._obs_counts.detach().cpu().numpy()[mask]
                centroids, extends, similarity_max_list, points, obs_max_list = find_clusters(
                    points.detach().cpu().numpy(), alignments, obs=obs_ids
                )
                output = [centroids, extends, similarity_max_list, points, obs_max_list]
            else:
                centroids, extends, similarity_max_list, points = find_clusters(
                    points.detach().cpu().numpy(), alignments, obs=None
                )
                output = [centroids, extends, similarity_max_list, points]

            debug_text += (
                "### - Found " + str(len(centroids)) + " instances that might be target object.\n"
            )
            if debug:
                output.append(debug_text)

            return output


def find_clusters(vertices: np.ndarray, similarity: np.ndarray, obs=None):
    # Calculate the number of top values directly
    top_positions = vertices
    # top_values = probability_over_all_points[top_indices].flatten()

    # Apply DBSCAN clustering
    dbscan = DBSCAN(eps=0.25, min_samples=5)
    clusters = dbscan.fit(top_positions)
    labels = clusters.labels_

    # Initialize empty lists to store centroids and extends of each cluster
    centroids = []
    extends = []
    similarity_list = []
    points = []
    obs_max_list = []

    for cluster_id in set(labels):
        if cluster_id == -1:  # Ignore noise
            continue

        members = top_positions[labels == cluster_id]
        centroid = np.mean(members, axis=0)

        similarity_values = similarity[labels == cluster_id]
        simiarity = np.max(similarity_values)

        if obs is not None:
            obs_values = obs[labels == cluster_id]
            obs_max = np.max(obs_values)

        sx = np.max(members[:, 0]) - np.min(members[:, 0])
        sy = np.max(members[:, 1]) - np.min(members[:, 1])
        sz = np.max(members[:, 2]) - np.min(members[:, 2])

        # Append centroid and extends to the lists
        centroids.append(centroid)
        extends.append((sx, sy, sz))
        similarity_list.append(simiarity)
        points.append(members)
        if obs is not None:
            obs_max_list.append(obs_max)

    if obs is not None:
        return centroids, extends, similarity_list, points, obs_max_list
    else:
        return centroids, extends, similarity_list, points
