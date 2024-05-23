from typing import List

import numpy as np
import torch

from stretch.core.parameters import Parameters
from stretch.mapping.instance import Instance


class SceneGraph:
    """Compute a very simple scene graph. Use it to extract relationships between instances."""

    def __init__(self, parameters: Parameters, instances: List[Instance]):
        self.parameters = parameters
        self.instances = instances
        self.relationships = []
        self.update(instances)

    def update(self, instances):
        """Extract pairwise symbolic spatial relationship between instances using heurisitcs"""
        self.relationships = []
        self.instances = instances
        for idx_a, ins_a in enumerate(instances):
            for idx_b, ins_b in enumerate(instances):
                if idx_a == idx_b:
                    continue
                # TODO: add "on", "in" ... relationships
                if (
                    self.near(ins_a.global_id, ins_b.global_id)
                    and (ins_b.global_id, ins_a.global_id, "near") not in self.relationships
                ):
                    self.relationships.append((ins_a.global_id, ins_b.global_id, "near"))

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

    def get_relationships(self, debug: bool = False):
        # show symbolic relationships
        if debug:
            for idx_a, idx_b, rel in self.relationships:
                print(idx_a, idx_b, rel)

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
                plt.show()
        # Return the detected relationships in list form
        return self.relationships

    def near(self, ins_a, ins_b):
        dist = torch.pairwise_distance(
            self.get_ins_center_pos(ins_a), self.get_ins_center_pos(ins_b)
        ).item()
        if dist < self.parameters["max_near_distance"]:
            return True
        return False

    def on(self, ins_a, ins_b):
        if (
            self.near(ins_a, ins_b)
            and self.get_ins_center_pos(ins_a)[2] > self.get_ins_center_pos(ins_b)[2]
        ):
            return True
        return False
