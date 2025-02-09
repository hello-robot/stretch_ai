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
from itertools import chain

import cv2
import imageio
import networkx as nx
import numpy as np
import torch

# from networkx.readwrite import json_graph
from openai import OpenAI
from PIL import Image
from pydantic import BaseModel
from transformers import AutoModel, AutoProcessor

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
        output_path,
        scene_graph,
        robot,
        rr_logger=None,
        device="cpu",
        clean_ques_ans=" ",
        enrich_object_labels=None,
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
        self._detector_path = output_path / "detector"
        self._sg_path = output_path / "filtered_dsg.json"
        self.scene_graph = scene_graph
        self._room_names = []

        os.makedirs(self._detector_path, exist_ok=True)

        self.rr_logger = rr_logger
        self.thresh = 2.0
        self.choose_final_image = True

        self.filter_out_objects = ["floor", "ceiling", "."]

        self.model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384").to(device)
        self.processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
        labels = self.enrich_object_labels.replace(".", "")
        exist = f"There is  {labels} in the scene."
        self.question_embed = self.processor(
            text=[clean_ques_ans], padding="max_length", return_tensors="pt"
        ).to(device)
        self.question_embed_labels = self.processor(
            text=[labels], padding="max_length", return_tensors="pt"
        ).to(device)
        self.question_embed_exist = self.processor(
            text=[exist], padding="max_length", return_tensors="pt"
        ).to(device)
        self.question_embed = self.question_embed_labels.copy()
        self.question_embed["input_ids"] = (
            (self.question_embed_labels["input_ids"] + self.question_embed_exist["input_ids"]) / 2.0
        ).to(self.question_embed_labels["input_ids"].dtype)

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

    def _build_sg_from_hydra_graph(self):
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
            attr["position"] = torch.mean(instance.pointcloud, dim=0)
            attr["name"] = instance.name + "_" + str(instance.global_id)
            attr["label"] = instance.name

            # object_node_positions.append(node.attributes.position)
            # bbox = node.attributes.bounding_box
            # bb_half_sizes.append(0.5 * bbox.dimensions)
            # bb_centroids.append(bbox.world_P_center)
            # bb_mat3x3.append(bbox.world_R_center)
            # bb_labels.append(node.attributes.name)
            # bb_colors.append(node.attributes.color)

            # if node_name in self.filter_out_objects:
            #     continue
            # self.filtered_obj_positions.append(node.attributes.position)
            # self.filtered_obj_ids.append(nodeid)
            # self._object_node_ids.append(nodeid)
            # self._object_node_names.append(node_name)

            self.filtered_netx_graph.add_nodes_from([(attr["name"], attr)])

        attr = {}
        xyt = self.robot.get_base_pose()
        attr["position"] = xyt
        attr["name"] = "Agent pose in (x, y, theta) format"
        self.filtered_netx_graph.add_nodes_from([("agent", attr)])

        self.curr_agent_id = "agent"
        self.curr_agent_pos = xyt

        # agent_ids, agent_cat_ids = [], []
        # for layer in self.pipeline.graph.dynamic_layers:
        #     for node in layer.nodes:
        #         if 'a' in node.id.category.lower():
        #             attr={}
        #             nodeid, node_type, node_name = self._get_node_properties(node)
        #             agent_cat_ids.append(int(node.id.category_id))
        #             agent_ids.append(nodeid)

        #             attr['position'] = list(node.attributes.position)
        #             attr['name'] = node_name
        #             attr['layer'] = node.layer
        #             attr['timestamp'] = float(node.timestamp/1e8)
        #             self.filtered_netx_graph.add_nodes_from([(nodeid, attr)])

        # if len(agent_cat_ids) > 0:
        #     self.curr_agent_id = agent_ids[np.argmax(agent_cat_ids)]
        #     self.curr_agent_pos = self.get_position_from_id(self.curr_agent_id)

        ## Adding other nodes

        # for node in self.pipeline.graph.nodes:
        #     attr={}
        #     nodeid, node_type, node_name = self._get_node_properties(node)
        #     attr['position'] = list(node.attributes.position)
        #     attr['name'] = node_name
        #     attr['layer'] = node.layer
        #     if self.rr_logger is not None:
        #         self.rr_logger.log_hydra_graph(is_node=True, nodeid=nodeid, node_type=node_type, node_pos_source=np.array(node.attributes.position))

        #     if node.id.category.lower() in ['o', 'r', 'b']:
        #         attr['label'] = node.attributes.semantic_label

        #     # Filtering
        #     if 'o' in node.id.category.lower():
        #         object_node_positions.append(node.attributes.position)
        #         bbox = node.attributes.bounding_box
        #         bb_half_sizes.append(0.5 * bbox.dimensions)
        #         bb_centroids.append(bbox.world_P_center)
        #         bb_mat3x3.append(bbox.world_R_center)
        #         bb_labels.append(node.attributes.name)
        #         bb_colors.append(node.attributes.color)

        #         if node_name in self.filter_out_objects:
        #             continue
        #         self.filtered_obj_positions.append(node.attributes.position)
        #         self.filtered_obj_ids.append(nodeid)
        #         self._object_node_ids.append(nodeid)
        #         self._object_node_names.append(node_name)

        #     if 'p' in node.id.category.lower():
        #         self._region_node_ids.append(nodeid)

        #     # if 'f' in node.id.category.lower():
        #     #     if self.is_relevant_frontier(np.array(attr['position']), self.curr_agent_pos)[0]:
        #     #         # self.rr_logger.log_hydra_graph(is_node=True, nodeid=nodeid, node_type='frontier_selected', node_pos_source=node.attributes.position)
        #     #         self._frontier_node_ids.append(nodeid)
        #     #         # DONT ADD FRONTIER OR PLACE NODES
        #     #         continue

        #     if 'r' in node.id.category.lower():
        #         self._room_ids.append(nodeid)

        #     self.filtered_netx_graph.add_nodes_from([(nodeid, attr)])

        self._room_names = self._room_ids.copy()
        self.bb_info = {
            "object_node_positions": object_node_positions,
            "bb_half_sizes": bb_half_sizes,
            "bb_centroids": bb_centroids,
            "bb_mat3x3": bb_mat3x3,
            "bb_labels": bb_labels,
            "bb_colors": bb_colors,
        }
        if self.rr_logger is not None:
            self.rr_logger.log_bb_data(self.bb_info)
        ## Adding edges
        for edge in chain(self.pipeline.graph.edges, self.pipeline.graph.dynamic_interlayer_edges):
            source_node = self.pipeline.graph.get_node(edge.source)
            sourceid, source_type, source_name = self._get_node_properties(source_node)

            target_node = self.pipeline.graph.get_node(edge.target)
            targetid, target_type, target_name = self._get_node_properties(target_node)
            edge_type = f"{source_type}-to-{target_type}"
            edgeid = f"{sourceid}-to-{targetid}"

            # Filtering scene graph
            if source_name in self.filter_out_objects or target_name in self.filter_out_objects:
                continue

            # if 'object' in source_type and 'object' in target_type: # Object->Object
            #     continue
            if "region" in source_type and "region" in target_type:  # Place->Place
                continue
            if (
                "frontier" in source_type or "frontier" in target_type
            ):  # ALL FRONTIERS for now, we add frontiers later
                continue
            if "agent" in source_type and "agent" in target_type:  # agent->agent
                continue

            if self.rr_logger is not None:
                self.rr_logger.log_hydra_graph(
                    is_node=False,
                    edge_type=edge_type,
                    edgeid=edgeid,
                    node_pos_source=np.array(source_node.attributes.position),
                    node_pos_target=np.array(target_node.attributes.position),
                )

            self.filtered_netx_graph.add_edges_from(
                [
                    (
                        sourceid,
                        targetid,
                        {"source_name": source_name, "target_name": target_name, "type": edge_type},
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

    def update(
        self, imgs_rgb=[], imgs_depth=None, intrinsics=None, extrinsics=None, frontier_nodes=[]
    ):

        self._build_sg_from_hydra_graph()
        self.update_frontier_nodes(frontier_nodes)

        self.save_best_image(imgs_rgb)

        self.add_room_labels_to_sg()

        if not self.include_regions:
            self.remove_region_nodes()

    def get_position_from_id(self, nodeid):
        return np.array(self.filtered_netx_graph.nodes[nodeid]["position"])

    def save_best_image(self, imgs_rgb, debug=False):

        img_idx = 0
        while (self.output_path / f"current_img_{img_idx}.png").exists():
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

            padding = "max_length"  # HuggingFace says SigLIP was trained on "max_length"
            imgs_embed = self.processor(
                images=sampled_images, return_tensors="pt", padding=padding
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**self.question_embed_labels, **imgs_embed)
            logits_per_text = outputs.logits_per_image  # this is the image-text similarity score
            probs = logits_per_text.softmax(
                dim=0
            ).squeeze()  # we can take the softmax to get the label probabilities

            probs, logits_per_text = (
                probs.detach().cpu().numpy(),
                logits_per_text.squeeze().detach().cpu().numpy(),
            )
            best = np.argmax(probs)
            top_k_indices = np.argsort(probs)[::-1][: self.topk]

            if debug:
                labeled_frames = []
                for idx in range(len(sampled_images)):
                    color_img = sampled_images[idx].copy()
                    label = f"{probs[idx]:.2f}"
                    if idx in top_k_indices:
                        label = label + f"_best{np.where(top_k_indices==idx)[0][0]}"
                    cv2.putText(
                        color_img,
                        str(label),
                        (20, 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 0),
                        1,
                        cv2.LINE_AA,
                    )
                    labeled_frames.append(color_img)

                imageio.mimsave(
                    self.output_path / f"images_with_clip_probs_{img_idx}.gif",
                    labeled_frames,
                    fps=0.5,
                )

            if self.save_image:
                rel_imgs = []
                for idx in range(len(top_k_indices)):
                    color_img = sampled_images[top_k_indices[idx]].copy()
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

                final_img.save(self.output_path / f"current_img_{img_idx}.png")
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
