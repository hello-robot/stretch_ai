import torch

from stretch.mapping.instance import Instance


class SceneGraph:
    """Compute a very simple scene graph. Use it to extract relationships between instances."""

    def __init__(self, instances):
        """Extract pairwise symbolic spatial relationship between instances using heurisitcs"""
        self.relationships = []
        for idx_a, ins_a in enumerate(instances):
            for idx_b, ins_b in enumerate(instances):
                if idx_a == idx_b:
                    continue
                # TODO: add "on", "in" ... relationships
                if (
                    self.near(ins_a.global_id, ins_b.global_id)
                    and (ins_b.global_id, ins_a.global_id, "near") not in relationships
                ):
                    self.relationships.append((ins_a.global_id, ins_b.global_id, "near"))

    def get_relationships(self, debug: bool = False):
        # show symbolic relationships
        if debug:
            for idx_a, idx_b, rel in self.relationships:
                import matplotlib.pyplot as plt

                plt.subplot(1, 2, 1)
                plt.imshow(
                    (
                        instances[idx_a].get_best_view().cropped_image
                        * instances[idx_a].get_best_view().mask
                        / 255.0
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
                plt.title("Instance A is " + rel)
                plt.axis("off")
                plt.subplot(1, 2, 2)
                plt.imshow(
                    (
                        instances[idx_b].get_best_view().cropped_image
                        * instances[idx_b].get_best_view().mask
                        / 255.0
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
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
