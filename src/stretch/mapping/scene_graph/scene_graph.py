# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from stretch.core.parameters import Parameters
from stretch.mapping.instance import Instance
from stretch.utils.memory import get_path_to_debug


class SceneGraph:
    """Compute a very simple scene graph. Use it to extract relationships between instances."""

    def __init__(self, parameters: Parameters, instances: List[Instance]):
        self.parameters = parameters
        self.instances = instances
        self.relationships: List[Tuple[int, int, str]] = []
        self.update(instances)

    def update(self, instances):
        """Extract pairwise symbolic spatial relationship between instances using heurisitcs"""
        self.relationships: List[Tuple[int, int, str]] = []
        self.instances = instances
        for idx_a, ins_a in enumerate(instances):
            for idx_b, ins_b in enumerate(instances):
                if idx_a == idx_b:
                    continue
                # TODO: add "on", "in" ... relationships
                # Add "near" relationship for if something is near something else - aka just within some distance threshold
                if (
                    self.near(ins_a.global_id, ins_b.global_id)
                    and (ins_b.global_id, ins_a.global_id, "near") not in self.relationships
                ):
                    self.relationships.append((ins_a.global_id, ins_b.global_id, "near"))

                # Add "on" relationship for if something is on top of something else
                if (self.on(ins_a.global_id, ins_b.global_id)) and (
                    ins_b.global_id,
                    ins_a.global_id,
                    "on",
                ) not in self.relationships:
                    self.relationships.append((ins_a.global_id, ins_b.global_id, "on"))
            # Add "on floor" relationship for if something is on the floor
            if self.on_floor(ins_a.global_id):
                self.relationships.append((ins_a.global_id, "floor", "on"))

    def get_matching_relations(
        self,
        id0: Optional[Union[int, str]],
        id1: Optional[Union[int, str]],
        relation: Optional[str],
    ) -> List[Tuple[int, int, str]]:
        """Get all relationships between two instances.

        Args:
            id0: The first instance id
            id1: The second instance id
            relation: The relationship between the two instances

        Returns:
            List of relationships in the form (idx_a, idx_b, relation)
        """
        if isinstance(id1, Instance):
            id1 = id1.global_id
        if isinstance(id0, Instance):
            id0 = id0.global_id
        return [
            rel
            for rel in self.relationships
            if (id0 is None or rel[0] == id0)
            and (id1 is None or rel[1] == id1)
            and (rel[2] == relation or relation is None)
        ]

    def get_ins_center_pos(self, idx: int):
        """Get the center of an instance based on point cloud"""
        return torch.mean(self.instances[idx].point_cloud, axis=0)

    def get_instance_image(self, idx: int) -> np.ndarray:
        """Get a viewable image from tensorized instances"""
        return (
            (
                self.instances[idx].get_best_view().cropped_image
                * self.instances[idx].get_best_view().mask
                / 255.0
            )
            .detach()
            .cpu()
            .numpy()
        )

    def get_relationships(self, debug: bool = False) -> List[Tuple[int, int, str]]:
        """Return the relationships between instances.

        Args:
            debug: If True, show the relationships in a matplotlib window

        Returns:
            List of relationships in the form (idx_a, idx_b, relation)
        """
        # show symbolic relationships
        if debug:
            for idx_a, idx_b, rel in self.relationships:
                print(idx_a, idx_b, rel)

                if idx_b == "floor":
                    img_a = self.get_instance_image(idx_a)
                    img_b = np.zeros_like(img_a)
                else:
                    img_a = self.get_instance_image(idx_a)
                    img_b = self.get_instance_image(idx_b)

                import matplotlib

                matplotlib.use("TkAgg")
                import matplotlib.pyplot as plt

                plt.subplot(1, 2, 1)
                plt.imshow(img_a)
                plt.title("Instance A is " + rel)
                plt.axis("off")
                plt.subplot(1, 2, 2)
                plt.imshow(img_b)
                plt.title("Instance B")
                plt.axis("off")
                # plt.show()
                plt.savefig(get_path_to_debug(f"scene_graph_{idx_a}_{idx_b}_{rel}.png"))

        # Return the detected relationships in list form
        return self.relationships

    def near(self, ins_a, ins_b):
        dist = torch.pairwise_distance(
            self.get_ins_center_pos(ins_a), self.get_ins_center_pos(ins_b)
        ).item()
        if dist < self.parameters["scene_graph"]["max_near_distance"]:
            return True
        return False

    def on(self, ins_a, ins_b):
        """On is defined as near and above, within some tolerance"""
        if self.near(ins_a, ins_b):
            z_dist = self.get_ins_center_pos(ins_a)[2] - self.get_ins_center_pos(ins_b)[2]
            if (
                z_dist < self.parameters["scene_graph"]["max_on_height"]
                and z_dist > self.parameters["scene_graph"]["min_on_height"]
            ):
                return True
        return False

    def on_floor(self, ins_a):
        """Check if an instance is on the floor"""
        pos = self.get_ins_center_pos(ins_a)
        if pos[2] < self.parameters["scene_graph"]["max_on_height"] and pos[2] > 0:
            return True
        return False
