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
from typing import List

import cv2
import networkx as nx
import numpy as np
import torch

# from networkx.readwrite import json_graph
from openai import OpenAI
from PIL import Image
from pydantic import BaseModel

# from transformers import AutoModel, AutoProcessor
from stretch.perception.encoders.siglip_encoder import SiglipEncoder

# from itertools import chain


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
        rr_logger=None,
        device: str = "cuda",
        clean_ques_ans=" ",
        enrich_object_labels="table",
        cache_size: int = 100,
    ):
        self.robot = robot
        self.topk = 2
        self.img_subsample_freq = 1
        self.device = device
        self.enrich_rooms = True
        self.enrich_object_labels = enrich_object_labels

        self.save_image = True
        self.include_regions = False
        self.enrich_frontiers = False

        self.output_path = output_path
        self.scene_graph = scene_graph
        self._room_names: List[str] = []

        self.rr_logger = rr_logger
        self.thresh = 2.0
        self.choose_final_image = True

        self.filter_out_objects = ["floor", "ceiling", "."]

        # self.model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384").to(device)
        # self.processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
        self.encoder = SiglipEncoder(version="so400m", device=device, normalize=True)
        labels = self.enrich_object_labels.replace(".", "")
        exist = f"There is  {labels} in the scene."
        with torch.no_grad():
            self.text_embed = (
                self.encoder.encode_text(labels) + self.encoder.encode_text(exist) / 2.0
            ).to(device)
        # self.question_embed = self.processor(
        #     text=[clean_ques_ans], padding="max_length", return_tensors="pt"
        # ).to(device)
        # self.question_embed_labels = self.processor(
        #     text=[labels], padding="max_length", return_tensors="pt"
        # ).to(device)
        # self.question_embed_exist = self.processor(
        #     text=[exist], padding="max_length", return_tensors="pt"
        # ).to(device)
        # self.question_embed = self.question_embed_labels.copy()
        # self.question_embed["input_ids"] = (
        #     (self.question_embed_labels["input_ids"] + self.question_embed_exist["input_ids"]) / 2.0
        # ).to(self.question_embed_labels["input_ids"].dtype)

        # For computing img embeds
        self.imgs_embed = None
        self.imgs_rgb = None
        self.cache_size = cache_size

    @property
    def scene_graph_str(self):
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
        self.filtered_obj_positions, self.filtered_obj_ids = [], []

        for instance in self.scene_graph.instances:
            attr = {}
            attr["position"] = torch.mean(instance.point_cloud, dim=0).tolist()
            attr["name"] = instance.name + "_" + str(instance.global_id)
            attr["label"] = instance.name

            # object_node_positions.append(attr["position"])
            # bbox = node.attributes.bounding_box
            # bb_half_sizes.append(0.5 * bbox.dimensions)
            # bb_centroids.append(bbox.world_P_center)
            # bb_mat3x3.append(bbox.world_R_center)
            # bb_labels.append(node.attributes.name)
            # bb_colors.append(node.attributes.color)

            if instance.name in self.filter_out_objects:
                continue
            self.filtered_obj_positions.append(attr["position"])
            self.filtered_obj_ids.append(attr["name"])
            self._object_node_ids.append(attr["name"])
            self._object_node_names.append(instance.name)

            self.filtered_netx_graph.add_nodes_from([(instance.global_id, attr)])

        attr = {}
        xyt = self.robot.get_base_pose()
        attr["position"] = torch.Tensor(xyt).tolist()
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

    def update_frontier_nodes(self, frontier_nodes):
        if len(frontier_nodes) > 0 and len(self.filtered_obj_positions) > 0:
            self.filtered_obj_positions = np.array(self.filtered_obj_positions)
            self.filtered_obj_ids = np.array(self.filtered_obj_ids)
            self._frontier_node_ids = []
            for i in range(frontier_nodes.shape[0]):
                attr = {}
                attr["position"] = list(frontier_nodes[i])
                attr["name"] = "frontier"
                attr["layer"] = 2
                nodeid = f"frontier_{i}"
                self._frontier_node_ids.append(nodeid)
                self.filtered_netx_graph.add_nodes_from([(nodeid, attr)])

                dist = np.linalg.norm(
                    (np.array(frontier_nodes[i]) - self.filtered_obj_positions), axis=1
                )
                relevant_objs = dist < self.thresh
                relevent_node_ids = self.filtered_obj_ids[relevant_objs]
                relevant_obj_pos = self.filtered_obj_positions[relevant_objs]

                if self.enrich_frontiers:
                    edge_type = "frontier-to-object"

                    for obj_id, obj_pos in zip(relevent_node_ids, relevant_obj_pos):
                        edgeid = f"{nodeid}-to-{obj_id}"

                        self.filtered_netx_graph.add_edges_from(
                            [
                                (
                                    nodeid,
                                    obj_id,
                                    {
                                        "source_name": "frontier",
                                        "target_name": "object",
                                        "type": edge_type,
                                    },
                                )
                            ]
                        )
                        if self.rr_logger is not None:
                            self.rr_logger.log_hydra_graph(
                                is_node=False,
                                edge_type=edge_type,
                                edgeid=edgeid,
                                node_pos_source=frontier_nodes[i],
                                node_pos_target=obj_pos,
                            )

    def add_room_labels_to_sg(self):
        self._room_names = []
        if len(self._room_ids) > 0:
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
        else:
            # If no room nodes exist, add room_0 to graph and edges to regions
            self._room_ids = ["room_0"]
            object_names = np.unique(
                [
                    self.filtered_netx_graph.nodes[object_id]["name"]
                    for object_id in self._object_node_ids
                ]
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
            print(f" ======== time for room enrichment: {time.time()-start}")

            # Add node to graph
            attr = {"name": completion.choices[0].message.parsed.room.value, "layer": 4}
            self.filtered_netx_graph.add_nodes_from([("room_0", attr)])
            self._room_names.append(completion.choices[0].message.parsed.room.value)

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

    # def _get_node_properties(self, node):
    #     # print(f"layer: {node.layer}. Category: {node.id.category.lower()}{node.id.category_id}. Active Frontier: {node.attributes.active_frontier}")
    #     if 'p' in node.id.category.lower():
    #         nodeid = f'region_{node.id.category_id}'
    #         node_type = 'region'
    #         node_name = 'region'
    #     if 'f' in node.id.category.lower():
    #         nodeid = f'frontier_{node.id.category_id}'
    #         node_type = 'frontier'
    #         node_name = 'frontier'
    #     if 'o' in node.id.category.lower():
    #         nodeid = f'object_{node.id.category_id}'
    #         node_type = 'object'
    #         node_name = node.attributes.name
    #     if 'r' in node.id.category.lower():
    #         nodeid = f'room_{node.id.category_id}'
    #         node_type = 'room'
    #         node_name = 'room'
    #     if 'b' in node.id.category.lower():
    #         nodeid = f'building_{node.id.category_id}'
    #         node_type = 'building'
    #         node_name = 'building'
    #     if 'a' in node.id.category.lower():
    #         nodeid = f'agent_{node.id.category_id}'
    #         node_type = 'agent'
    #         node_name = 'agent'
    #     return nodeid, node_type, node_name

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

    def update(self, imgs_rgb, frontier_nodes=[]):

        self._build_sg_from_hydra_graph()
        self.update_frontier_nodes(frontier_nodes)

        self.save_best_image(imgs_rgb)

        # self.add_room_labels_to_sg()

        # if not self.include_regions:
        #     self.remove_region_nodes()

    def get_position_from_id(self, nodeid):
        return np.array(self.filtered_netx_graph.nodes[nodeid]["position"])

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

            # logits = self.encoder.compute_score(self.imgs_embed, self.text_embed)
            logits = torch.mean(
                self.imgs_embed @ self.text_embed.reshape(-1, self.imgs_embed.shape[-1]).T, dim=-1
            )
            top_k_indices = torch.argsort(logits, descending=True)[: self.topk]

            # with torch.no_grad():
            #     outputs = self.model(**self.question_embed_labels, **self.imgs_embed)
            # logits_per_text = outputs.logits_per_image  # this is the image-text similarity score
            # probs = logits_per_text.softmax(
            #     dim=0
            # ).squeeze()  # we can take the softmax to get the label probabilities

            # probs, logits_per_text = (
            #     probs.detach().cpu().numpy(),
            #     logits_per_text.squeeze().detach().cpu().numpy(),
            # )
            # best = np.argmax(probs)
            # top_k_indices = np.argsort(probs)[::-1][: self.topk]

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
