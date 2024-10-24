# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# This file contains the UpdateOperation class, which is responsible for updating the world model
# and finding objects in the environment. It is a subclass of ManagedOperation.
import time
from typing import Optional

import numpy as np

from stretch.agent.base import ManagedOperation


class UpdateOperation(ManagedOperation):

    show_instances_detected: bool = False
    show_map_so_far: bool = False
    clear_voxel_map: bool = False
    move_head: Optional[bool] = None
    target_object: str = "cup"
    match_method: str = "feature"
    arm_height: float = 0.4
    on_floor_only: bool = False

    def set_target_object_class(self, object_class: str):
        """Set the target object class for the operation.

        Args:
            object_class (str): The object class to set as the target.
        """
        self.warn(f"Overwriting target object class from {self.object_class} to {object_class}.")  # type: ignore
        self.object_class = object_class

    def can_start(self):
        return True

    def configure(
        self,
        move_head=False,
        show_instances_detected=False,
        show_map_so_far=False,
        clear_voxel_map=False,
        target_object: str = "cup",
        match_method: str = "feature",
        arm_height: float = 0.4,
    ):
        """Configure the operation with the given parameters."""
        self.move_head = move_head
        self.show_instances_detected = show_instances_detected
        self.show_map_so_far = show_map_so_far
        self.clear_voxel_map = clear_voxel_map
        self.target_object = target_object
        self.match_method = match_method
        self.arm_height = arm_height
        if self.match_method not in ["class", "feature"]:
            raise ValueError(f"Unknown match method {self.match_method}.")
        print("---- CONFIGURING UPDATE OPERATION ----")
        print("Move head is set to", self.move_head)
        print("Show instances detected is set to", self.show_instances_detected)
        print("Show map so far is set to", self.show_map_so_far)
        print("Clear voxel map is set to", self.clear_voxel_map)
        print("Target object is set to", self.target_object)
        print("Match method is set to", self.match_method)
        print("Arm height is set to", self.arm_height)
        print("--------------------------------------")

    def run(self):
        """Run the operation."""
        self.intro("Updating the world model.")
        if self.clear_voxel_map:
            self.agent.reset()
        if not self.robot.in_manipulation_mode():
            self.warn("Robot is not in manipulation mode. Moving to manip posture.")
            self.robot.move_to_manip_posture()
            time.sleep(2.0)
        self.robot.arm_to([0.0, self.arm_height, 0.05, 0, -np.pi / 4, 0], blocking=True)
        xyt = self.robot.get_base_pose()

        # Now update the world
        self.update(move_head=self.move_head)

        # Delete observations near us, since they contain the arm!!
        self.agent.voxel_map.delete_obstacles(point=xyt[:2], radius=0.7, force_update=True)

        # Notify and move the arm back to normal. Showing the map is optional.
        print(f"So far we have found: {len(self.agent.voxel_map.instances)} objects.")
        self.robot.arm_to([0.0, self.arm_height, 0.05, 0, -np.pi / 4, 0], blocking=True)

        if self.show_map_so_far:
            # This shows us what the robot has found so far
            xyt = self.robot.get_base_pose()
            self.agent.voxel_map.show(
                orig=np.zeros(3),
                xyt=xyt,
                footprint=self.robot_model.get_footprint(),
                planner_visuals=True,
            )

        if self.show_instances_detected:
            # Show the last instance image
            import matplotlib

            # TODO: why do we need to configure this every time
            matplotlib.use("TkAgg")
            import matplotlib.pyplot as plt

            plt.imshow(self.agent.voxel_map.observations[-1].instance)
            plt.show()

        # Describe the scene the robot is operating in
        scene_graph = self.agent.get_scene_graph()

        # Get the current location of the robot
        start = self.robot.get_base_pose()
        instances = self.agent.voxel_map.instances.get_instances()
        receptacle_options = []
        object_options = []
        dist_to_object = float("inf")

        # Find the object we want to manipulate
        if self.match_method == "class":
            instances = self.agent.voxel_map.instances.get_instances_by_class(self.target_object)
            scores = np.ones(len(instances))
        elif self.match_method == "feature":
            scores, instances = self.agent.get_instances_from_text(self.target_object)
            # self.agent.voxel_map.show(orig=np.zeros(3), xyt=start, footprint=self.robot_model.get_footprint(), planner_visuals=True)
        else:
            raise ValueError(f"Unknown match type {self.match_method}")

        if len(instances) == 0:
            self.warn(f"Could not find any instances of {self.target_object}.")

        print("Check explored instances for reachable receptacles:")
        for i, (score, instance) in enumerate(zip(scores, instances)):
            name = self.agent.semantic_sensor.get_class_name_for_id(instance.category_id)
            print(
                f" - Found instance {i} with name {name} and global id {instance.global_id}: score = {score}."
            )

            if self.show_instances_detected:
                self.show_instance(instance)

            if self.on_floor_only:
                relations = scene_graph.get_matching_relations(instance.global_id, "floor", "on")
                if len(relations) == 0:
                    # This may or may not be what we want, but it certainly is not on the floor
                    continue

            object_options.append(instance)
            dist = np.linalg.norm(instance.point_cloud.mean(axis=0).cpu().numpy()[:2] - start[:2])
            if dist < 1.0:
                self.agent.current_object = instance
                return

            plan = self.plan_to_instance_for_manipulation(instance, start=start)
            if plan.success:
                print(f" - Found a reachable object at {instance.get_best_view().get_pose()}.")
                if dist < dist_to_object:
                    print(
                        f" - This object is closer than the previous one: {dist} < {dist_to_object}."
                    )
                    self.agent.current_object = instance
                    dist_to_object = dist
            else:
                self.warn(f" - Found an object of class {name} but could not reach it.")

    def was_successful(self):
        """We're just taking an image so this is always going to be a success"""
        return self.agent.current_object is not None
