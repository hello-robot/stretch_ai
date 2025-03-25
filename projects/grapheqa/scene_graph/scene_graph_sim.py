# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import json
import os
import time
from enum import Enum
from typing import List, Union

import cv2
import networkx as nx
import numpy as np
import torch

# from networkx.readwrite import json_graph
from openai import OpenAI
from PIL import Image
from pydantic import BaseModel

from stretch.perception.captioners.gemma_captioner import GemmaCaptioner
from stretch.perception.captioners.internvl_captioner import InternvlCaptioner
from stretch.perception.captioners.openai_captioner import OpenaiCaptioner
from stretch.perception.captioners.paligemma_captioner import PaligemmaCaptioner
from stretch.perception.captioners.qwen_captioner import QwenCaptioner

Grounded_Captioner = [
    QwenCaptioner,
    PaligemmaCaptioner,
    GemmaCaptioner,
    InternvlCaptioner,
    OpenaiCaptioner,
]


if "OPENAI_API_KEY" in os.environ:
    client = OpenAI()
else:
    print("GPT token has not been set up yet!")


class Rooms(str, Enum):
    bedroom = "bedroom"
    bathroom = "bathroom"
    living_room = "living room"
    kitchen = "kitchen"
    lobby = "lobby"
    dining_room = "dining room"
    patio = "patio"
    closet = "closet"
    study = "study room"
    staircase = "staircase"
    porch = "porch"
    laboratory = "laboratory"
    office = "office"
    workshop = "workshop"
    garage = "garage"


class Room_response(BaseModel):
    explanation: str
    room: Rooms


class SceneGraphSim:
    def __init__(
        self,
        output_path: str,
        scene_graph,
        robot,
        captioner=None,
        rr_logger=None,
        device: str = "cuda",
        clean_ques_ans=" ",
        cache_size: int = 100,
        encoder=None,
        use_class_labels: bool = False,
    ):
        self.robot = robot
        self.topk = 2
        self.img_subsample_freq = 1
        self.device = device
        # TODO: remove hard coding
        self.enrich_rooms = False

        self.save_image = True
        self.include_regions = False
        # self.enrich_frontiers = True

        self.output_path = output_path
        self.scene_graph = scene_graph
        self._room_names: List[str] = []

        self.rr_logger = rr_logger
        # For constructing frontier nodes and connecting them with object nodes
        self.thresh = 1.5
        self.size_thresh = 0.1
        self.choose_final_image = False
        self.use_class_labels = use_class_labels

        self.filter_out_objects = ["floor", "ceiling", "wall", "."]

        # minimum size of bounding boxes for the instance to be added on the scene graph string
        self.min_area = 4000

        self.encoder = encoder
        if self.encoder is None:
            self.save_image = False
        # by default is "object", will be updated later
        self.enrich_object_labels: Union[List[str], str] = "object"
        labels = self.enrich_object_labels.replace(".", "")
        exist = f"There is  {labels} in the scene."
        with torch.no_grad():
            self.text_embeds = (
                (self.encoder.encode_text(labels) + self.encoder.encode_text(exist) / 2.0)
                .to(device)
                .unsqueeze(0)
            )
        self.captioner = captioner

        # For computing img embeds
        self.imgs_embed = None
        self.imgs_rgb = None
        self.cache_size = cache_size

        # For saving image
        self.current_step = 0
        self.images = None

    @property
    def scene_graph_str(self):
        print(json.dumps(nx.node_link_data(self.filtered_netx_graph)))
        return json.dumps(nx.node_link_data(self.filtered_netx_graph))

    @property
    def room_node_ids(self):
        return self._room_ids

    @property
    def room_node_names(self):
        return self._room_names

    @property
    def region_node_ids(self):
        return self._region_node_ids

    @property
    def frontier_node_ids(self):
        return self._frontier_node_ids

    @property
    def object_node_ids(self):
        return self._object_node_ids

    @property
    def object_node_names(self):
        return self._object_node_names

    def update_language_embedding(self, texts: Union[List[str], str]):
        """
        When we update self.enrich_object_labels, we call this function to update the corresponding embedding

        Args:
            - text: the new labels text
        """
        self.enrich_object_labels = [texts] if isinstance(texts, str) else texts
        assert len(self.enrich_object_labels) > 0, "enrich_object_labels cannot be none"
        self.text_embeds = []
        for enrich_object_label in self.enrich_object_labels:
            label = enrich_object_label.replace(".", "")
            with torch.no_grad():
                self.text_embeds.append(self.encoder.encode_text(label).to(self.device))
        self.text_embeds = torch.stack(self.text_embeds)

    def is_relevant_frontier(self, frontier_node_positions, agent_pos):
        frontier_node_positions = frontier_node_positions.reshape(-1, 3)
        thresh_low = agent_pos[2] - 0.75
        thresh_high = agent_pos[2] + 0.3
        in_plane = np.logical_and(
            (frontier_node_positions[:, 2] < thresh_high),
            (frontier_node_positions[:, 2] > thresh_low),
        )
        nearby = np.linalg.norm(frontier_node_positions - agent_pos, axis=-1) < 2.0
        return np.logical_and(in_plane, nearby)

    def _build_sg_from_hydra_graph(self, add_object_edges=False):
        start = time.time()
        self.filtered_netx_graph = nx.DiGraph()

        (
            self._room_ids,
            self._region_node_ids,
            self._frontier_node_ids,
            self._object_node_ids,
            self._object_node_names,
        ) = ([], [], [], [], [])

        object_node_positions, bb_half_sizes, bb_centroids, bb_mat3x3, bb_labels, bb_colors = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        self.filtered_obj_positions, self.filtered_obj_ids, self.filtered_obj_sizes = [], [], []

        for instance in self.scene_graph.instances:
            if instance.name in self.filter_out_objects:
                continue
            attr = {}
            attr["position"] = torch.mean(instance.point_cloud, dim=0).tolist()
            # round up to prevent the scene graph str from being too long
            attr["position"] = [round(coord, 3) for coord in attr["position"]]
            best_view = instance.get_best_view()
            area = (best_view.bbox[1, 1] - best_view.bbox[0, 1]) * (
                best_view.bbox[1, 0] - best_view.bbox[0, 0]
            )
            if area < self.min_area:
                continue
            if self.captioner is None:
                attr["name"] = instance.name + "_" + str(instance.global_id)
            elif best_view.text_description is None:
                # Qwen captioner takes a complete image plus a bounding box
                if type(self.captioner) in Grounded_Captioner:
                    bbox = []
                    bbox.append(int(best_view.bbox[0, 1].item()) - 10)
                    bbox.append(int(best_view.bbox[0, 0].item()) - 10)
                    bbox.append(int(best_view.bbox[1, 1].item()) + 10)
                    bbox.append(int(best_view.bbox[1, 0].item()) + 10)
                    try:
                        best_view.text_description = self.captioner.caption_image(
                            best_view.cropped_image.to(dtype=torch.uint8), bbox
                        )
                    except:
                        print("Caption cannot be generated!")
                        continue
                else:
                    best_view.text_description = self.captioner.caption_image(
                        best_view.cropped_image.to(dtype=torch.uint8)
                    )
                attr["name"] = best_view.text_description
            else:
                attr["name"] = best_view.text_description

            # attr["label"] = "object_" + str(instance.global_id)
            # bounds is a (3 x 2) mins and max
            # attr["bbox"] = instance.bounds.tolist()
            size = round(
                torch.prod(torch.abs(instance.bounds[:, 0] - instance.bounds[:, 1])).item(), 4
            )
            # attr["size"] = size
            # node_id = instance.name + "_" + str(instance.global_id)
            node_id = "object_" + str(instance.global_id)

            # object_node_positions.append(attr["position"])
            # bbox = node.attributes.bounding_box
            # bb_half_sizes.append(0.5 * bbox.dimensions)
            # bb_centroids.append(bbox.world_P_center)
            # bb_mat3x3.append(bbox.world_R_center)
            # bb_labels.append(node.attributes.name)
            # bb_colors.append(node.attributes.color)

            self.filtered_obj_positions.append(attr["position"])
            self.filtered_obj_sizes.append(size)
            self.filtered_obj_ids.append(node_id)
            self._object_node_ids.append(node_id)
            self._object_node_names.append(instance.name)

            self.filtered_netx_graph.add_nodes_from([(node_id, attr)])

        attr = {}
        xyt = self.robot.get_base_pose()
        attr["position"] = [round(coord, 3) for coord in torch.Tensor(xyt).tolist()]
        attr["name"] = "Agent pose in (x, y, theta) format"
        self.filtered_netx_graph.add_nodes_from([("agent", attr)])

        self.curr_agent_id = "agent"
        self.curr_agent_pos = xyt
        self._room_names = self._room_ids.copy()

        ## Adding edges

        if add_object_edges:
            for edge in self.scene_graph.get_relationships():
                idx_a, idx_b, rel = edge
                if idx_b == "floor":
                    continue
                instance_a = self.scene_graph.instances[idx_a]
                instance_b = self.scene_graph.instances[idx_b]
                if (
                    instance_a.name in self.filter_out_objects
                    or instance_b in self.filter_out_objects
                ):
                    continue
                sourceid = instance_a.name + "_" + str(instance_a.global_id)
                targetid = instance_b.name + "_" + str(instance_b.global_id)
                edge_type = "object-to-object"
                edge_id = f"{sourceid}-to-{targetid}"
                self.filtered_netx_graph.add_edges_from(
                    [
                        (
                            sourceid,
                            targetid,
                            {
                                "source_name": instance_a.name,
                                "target_name": instance_b.name,
                                "type": edge_type,
                            },
                        )
                    ]
                )

        print(f"===========time taken for Captioning and SG building: {time.time()-start}")

    def update_frontier_nodes(self, frontier_nodes):
        if len(frontier_nodes) > 0 and len(self.filtered_obj_positions) > 0:
            self.filtered_obj_positions = np.array(self.filtered_obj_positions)
            self.filtered_obj_ids = np.array(self.filtered_obj_ids)
            self._frontier_node_ids = []
            for i in range(frontier_nodes.shape[0]):
                attr = {}
                attr["position"] = list(frontier_nodes[i])
                attr["name"] = "frontier"
                # attr["layer"] = 2
                nodeid = f"frontier_{i}"

                dist = np.linalg.norm(
                    (np.array(frontier_nodes[i]) - self.filtered_obj_positions), axis=1
                )
                relevant_objs = (dist < self.thresh) & (
                    np.array(self.filtered_obj_sizes) > self.size_thresh
                )
                relevent_node_ids = self.filtered_obj_ids[relevant_objs]
                relevant_obj_pos = self.filtered_obj_positions[relevant_objs]

                description = "Nearby objects: "
                for nearby_object_id in relevent_node_ids:
                    description += nearby_object_id + "; "
                attr["name"] = description

                self._frontier_node_ids.append(nodeid)
                self.filtered_netx_graph.add_nodes_from([(nodeid, attr)])

                # if self.enrich_frontiers:
                #     edge_type = "frontier-to-object"

                #     for obj_id, obj_pos in zip(relevent_node_ids, relevant_obj_pos):
                #         edgeid = f"{nodeid}-to-{obj_id}"

                #         self.filtered_netx_graph.add_edges_from(
                #             [
                #                 (
                #                     nodeid,
                #                     obj_id,
                #                     {
                #                         "source_name": "frontier",
                #                         "target_name": "object",
                #                         "type": edge_type,
                #                     },
                #                 )
                #             ]
                #         )

    def add_room_labels_to_sg(self):
        self._room_names = []
        if len(self._room_ids) > 0 and self.enrich_rooms:
            for room_id in self._room_ids:
                place_ids = [
                    place_id
                    for place_id in self.filtered_netx_graph.successors(room_id)
                    if "room" not in place_id
                ]  # ignore room->room
                object_ids = [
                    object_id
                    for place_id in place_ids
                    for object_id in self.filtered_netx_graph.successors(place_id)
                    if "agent" not in object_id
                ]  # ignore place->agent
                object_names = np.unique(
                    [self.filtered_netx_graph.nodes[object_id]["name"] for object_id in object_ids]
                )

                start = time.time()
                completion = client.beta.chat.completions.parse(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "user",
                            "content": f"Given the list of objects: {object_names}. Which room are these objects most likely found in? Keep explanation very brief.",
                        }
                    ],
                    response_format=Room_response,
                )
                print(f" ======== time for room {room_id} enrichment: {time.time()-start}")
                self.filtered_netx_graph.nodes[room_id]["name"] = completion.choices[
                    0
                ].message.parsed.room.value
                self._room_names.append(completion.choices[0].message.parsed.room.value)
        elif len(self._room_ids) == 0:
            # If no room nodes exist, add room_0 to graph and edges to regions
            self._room_ids = ["room_0"]
            object_names = np.unique(
                [
                    self.filtered_netx_graph.nodes[object_id]["name"]
                    for object_id in self._object_node_ids
                ]
            )
            if self.enrich_rooms:
                start = time.time()
                completion = client.beta.chat.completions.parse(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "user",
                            "content": f"Given the list of objects: {object_names}. Which room are these objects most likely found in? Keep explanation very brief.",
                        }
                    ],
                    response_format=Room_response,
                )
                print(f" ======== time for room enrichment: {time.time()-start}")

                # Add node to graph
                attr = {"name": completion.choices[0].message.parsed.room.value, "layer": 4}
                self._room_names.append(completion.choices[0].message.parsed.room.value)
            else:
                attr = {"name": "room_0", "layer": 4}
                self._room_names.append("room_0")
            self.filtered_netx_graph.add_nodes_from([("room_0", attr)])

            # Add edges from room to region
            edge_type = "room-to-region"
            for reg_id in self._region_node_ids:
                self.filtered_netx_graph.add_edges_from(
                    [
                        (
                            "room_0",
                            reg_id,
                            {"source_name": "room", "target_name": "region", "type": edge_type},
                        )
                    ]
                )

    def get_current_semantic_state_str(self):
        agent_pos = self.filtered_netx_graph.nodes[self.curr_agent_id]["position"]
        agent_loc_str = (
            f"The agent is currently at node {self.curr_agent_id} at position {agent_pos}"
        )
        if self.include_regions:
            agent_place_ids = [
                place_id for place_id in self.filtered_netx_graph.predecessors(self.curr_agent_id)
            ]
            room_id = [
                room_id for room_id in self.filtered_netx_graph.predecessors(agent_place_ids[0])
            ]
        else:
            room_id = [
                room_id for room_id in self.filtered_netx_graph.predecessors(self.curr_agent_id)
            ]

        room_str = ""
        if len(room_id) > 0:
            room_name = self.filtered_netx_graph.nodes[room_id[0]]["name"]
            if room_name != "room":
                room_str = f" at room node: {room_id[0]} with name {room_name}"
        return f"{agent_loc_str} {room_str}"

    def update(self, frontier_nodes=[], imgs_rgb=[]):

        self._build_sg_from_hydra_graph()
        self.update_frontier_nodes(frontier_nodes)

        self.save_best_images_with_scene_graph()
        # self.save_best_image(imgs_rgb=imgs_rgb)

        self.add_room_labels_to_sg()

        # if not self.include_regions:
        #     self.remove_region_nodes()

    def get_position_from_id(self, nodeid):
        return np.array(self.filtered_netx_graph.nodes[nodeid]["position"])

    def save_best_images_with_scene_graph(self):
        """
        Given smenatic features stored, return most relevant images
        """
        start = time.time()
        image_embedding_list = []
        image_list = []
        instance_id_list = []

        # Gather all image list
        for instance in self.scene_graph.instances:
            image_embedding = instance.get_image_embedding(aggregation_method="mean").reshape(-1)
            instance_view = instance.get_best_view()
            image_embedding_list.append(image_embedding)
            image_list.append(instance_view.cropped_image)
            instance_id_list.append(instance.global_id)
        image_embeddings = torch.stack(image_embedding_list).to(self.device)

        similarity_matrix = (
            image_embeddings @ self.text_embeds.reshape(-1, image_embeddings.shape[-1]).T
        )
        top_k_indices = torch.argsort(similarity_matrix, dim=0, descending=True)

        self.images = {}
        for (i, img_indices) in enumerate(top_k_indices.transpose(1, 0)):
            rel_imgs = []
            label = self.enrich_object_labels[i]
            object_ids = ""
            # For each text query, find the most relevant topk images
            for idx in range(len(img_indices)):
                color_img = np.array(image_list[img_indices[idx].item()]).copy()
                cv2.putText(
                    color_img,
                    str(f"Image {idx+1}"),
                    (20, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

                # check whether this image has already been added. This process forces different images will be added.
                added = False
                for i in rel_imgs:
                    if np.all((np.abs(i - color_img) <= 10).reshape(-1)):
                        added = True
                if not added:
                    rel_imgs.append(color_img)
                    # if "object_" + str(instance_id_list[img_indices[idx].item()]) in self._object_node_ids:
                    object_ids += "object_" + str(instance_id_list[img_indices[idx].item()]) + ","
                if len(rel_imgs) >= self.topk:
                    break

            final_img = Image.fromarray(np.concatenate(rel_imgs, axis=1).astype(np.uint8))
            final_img.save(self.output_path + f"/current_img_{self.current_step}_{label}.png")
            self.images[object_ids] = final_img
        self.current_step += 1
        print(f"===========time taken for SigLIP emb: {time.time()-start}")

    def save_best_image(self, imgs_rgb):

        img_idx = 0
        while os.path.exists(self.output_path + f"/current_img_{img_idx}.png"):
            img_idx += 1

        if len(imgs_rgb) > 0 and self.save_image:
            start = time.time()
            imgs_rgb = np.array(imgs_rgb)
            w, h = imgs_rgb[0].shape[0], imgs_rgb[0].shape[1]
            # Remove black images
            black_pixels_mask = np.all(imgs_rgb == 0, axis=-1)
            num_black_pixels = np.sum(black_pixels_mask, axis=(1, 2))
            useful_img_idxs = num_black_pixels < 0.3 * w * h
            useful_imgs = imgs_rgb[useful_img_idxs]
            sampled_images = useful_imgs[:: self.img_subsample_freq]

            # padding = "max_length"  # HuggingFace says SigLIP was trained on "max_length"
            # imgs_embed = self.processor(
            #     images=sampled_images, return_tensors="pt", padding=padding
            # ).to(self.device)
            with torch.no_grad():
                imgs_embed = self.encoder.encode_image(sampled_images).to(self.device)

            if self.imgs_embed is None:
                self.imgs_embed = imgs_embed
            else:
                self.imgs_embed = torch.cat((self.imgs_embed, imgs_embed), dim=0)
            if self.imgs_rgb is None:
                self.imgs_rgb = sampled_images
            else:
                self.imgs_rgb = np.concatenate((self.imgs_rgb, sampled_images), axis=0)

            if len(self.imgs_rgb) > self.cache_size:
                self.imgs_rgb = self.imgs_rgb[-self.cache_size :, :]
                self.imgs_embed = self.imgs_embed[-self.cache_size :, :]

            logits = torch.mean(
                self.imgs_embed @ self.text_embeds.reshape(-1, self.imgs_embed.shape[-1]).T, dim=-1
            )
            top_k_indices = torch.argsort(logits, descending=True)[: self.topk]

            rel_imgs = []
            for idx in range(len(top_k_indices)):
                color_img = np.array(self.imgs_rgb)[top_k_indices[idx]].copy()
                cv2.putText(
                    color_img,
                    str(f"Image {idx+1}"),
                    (20, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )
                rel_imgs.append(color_img)
                # adding the last image
            if self.choose_final_image:
                color_img = imgs_rgb[-1].copy()
                cv2.putText(
                    color_img,
                    str(f"Image {len(top_k_indices)+1}"),
                    (20, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )
                rel_imgs.append(color_img)

            final_img = Image.fromarray(np.concatenate(rel_imgs, axis=1))

            final_img.save(self.output_path + f"/current_img_{img_idx}.png")
            print(f"===========time taken for CLIP/SigLIP emb: {time.time()-start}")

    def remove_close_positions(self, data, threshold):
        # Sort the list by confidence in descending order
        data_sorted = sorted(data, key=lambda x: x["confidence"], reverse=True)

        result = []
        for i, current in enumerate(data_sorted):
            # Check if current position is too close to any already included points
            too_close = False
            for chosen in result:
                # Compute the distance between positions
                distance = np.linalg.norm(np.array(current["pos"]) - np.array(chosen["pos"]))
                if distance < threshold:
                    too_close = True
                    break

            # Only keep if not too close to any already selected points
            if not too_close:
                result.append(current)

        return result

    def remove_region_nodes(self):
        # Identify and process each 'room' node
        for room_id in self.room_node_ids:
            # Find all 'region' nodes connected to this 'room'
            place_ids = [
                place_id
                for place_id in self.filtered_netx_graph.successors(room_id)
                if "room" not in place_id
            ]
            object_ids = [
                object_id
                for place_id in place_ids
                for object_id in self.filtered_netx_graph.successors(place_id)
                if "agent" not in object_id
            ]  # ignore place->agent
            agent_ids = [
                agent_id
                for place_id in place_ids
                for agent_id in self.filtered_netx_graph.successors(place_id)
                if "agent" in agent_id
            ]  # only place->agent

            # For each 'region' node, connect the 'room' directly to the 'object' children
            for object_id in object_ids:
                # Add edges from room to region
                edge_type = "room-to-object"
                self.filtered_netx_graph.add_edges_from(
                    [
                        (
                            room_id,
                            object_id,
                            {"source_name": "room", "target_name": "object", "type": edge_type},
                        )
                    ]
                )

            for agent_id in agent_ids:
                # Add edges from room to region
                edge_type = "room-to-agent"
                self.filtered_netx_graph.add_edges_from(
                    [
                        (
                            room_id,
                            agent_id,
                            {"source_name": "room", "target_name": "agent", "type": edge_type},
                        )
                    ]
                )
            self.filtered_netx_graph.remove_nodes_from(place_ids)
