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

import numpy as np

from stretch.agent.base import ManagedOperation


class UpdateOperation(ManagedOperation):

    show_instances_detected: bool = False
    show_map_so_far: bool = False
    clear_voxel_map: bool = False

    def set_target_object_class(self, object_class: str):
        self.warn(f"Overwriting target object class from {self.object_class} to {object_class}.")
        self.object_class = object_class

    def can_start(self):
        return True

    def run(self):
        self.intro("Updating the world model.")
        if self.clear_voxel_map:
            self.agent.reset()
        if not self.robot.in_manipulation_mode():
            self.warn("Robot is not in manipulation mode. Moving to manip posture.")
            self.robot.move_to_manip_posture()
            time.sleep(2.0)
        self.robot.arm_to([0.0, 0.4, 0.05, 0, -np.pi / 4, 0], blocking=True)
        xyt = self.robot.get_base_pose()
        # Now update the world
        self.update()
        # Delete observations near us, since they contain the arm!!
        self.agent.voxel_map.delete_obstacles(point=xyt[:2], radius=0.7)

        # Notify and move the arm back to normal. Showing the map is optional.
        print(f"So far we have found: {len(self.agent.voxel_map.instances)} objects.")
        self.robot.arm_to([0.0, 0.4, 0.05, 0, -np.pi / 4, 0], blocking=True)

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

        print("Check explored instances for reachable receptacles:")
        for i, instance in enumerate(instances):
            name = self.agent.semantic_sensor.get_class_name_for_id(instance.category_id)
            print(f" - Found instance {i} with name {name} and global id {instance.global_id}.")

            if self.show_instances_detected:
                self.show_instance(instance)

            # Find a box
            if "box" in name or "tray" in name:
                receptacle_options.append(instance)

                # Check to see if we can motion plan to box or not
                plan = self.plan_to_instance_for_manipulation(instance, start=start)
                if plan.success:
                    print(f" - Found a reachable box at {instance.get_best_view().get_pose()}.")
                    self.agent.current_receptacle = instance
                else:
                    self.warn(f" - Found a receptacle but could not reach it.")
            elif self.agent.target_object in name:
                relations = scene_graph.get_matching_relations(instance.global_id, "floor", "on")
                if len(relations) == 0:
                    # This may or may not be what we want, but it certainly is not on the floor
                    continue

                object_options.append(instance)
                dist = np.linalg.norm(
                    instance.point_cloud.mean(axis=0).cpu().numpy()[:2] - start[:2]
                )
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
