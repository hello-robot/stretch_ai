# Copyright (c) Hello Robot, Inc.
#
# This source code is licensed under the APACHE 2.0 license found in the
# LICENSE file in the root directory of this source tree.
# 
# Some code may be adapted from other open-source works with their respective licenses. Original
# licence information maybe found below, if so.


from typing import Optional

import numpy as np
from PIL import Image

from stretch.agent.base import ManagedOperation


class SearchForReceptacleOperation(ManagedOperation):
    """Find a place to put the objects we find on the floor. Will explore the map for a receptacle."""

    # For debugging
    show_map_so_far: bool = False
    show_instances_detected: bool = False

    def can_start(self) -> bool:
        self.attempt("will start searching for a receptacle on the floor.")
        return True

    def run(self) -> None:
        """Search for a receptacle on the floor."""

        # Update world map
        self.intro("Searching for a receptacle on the floor.")
        # Must move to nav before we can do anything
        self.robot.move_to_nav_posture()
        # Now update the world
        self.update()

        print(f"So far we have found: {len(self.manager.instance_memory)} objects.")

        if self.show_map_so_far:
            # This shows us what the robot has found so far
            xyt = self.robot.get_base_pose()
            self.agent.voxel_map.show(
                orig=np.zeros(3), xyt=xyt, footprint=self.robot_model.get_footprint()
            )

        if self.show_instances_detected:
            self.show_instance_segmentation_image()

        # Get the current location of the robot
        start = self.robot.get_base_pose()
        if not self.navigation_space.is_valid(start):
            self.error(
                "Robot is in an invalid configuration. It is probably too close to geometry, or localization has failed."
            )
            breakpoint()

        # Check to see if we have a receptacle in the map
        instances = self.manager.instance_memory.get_instances()
        print("Check explored instances for reachable receptacles:")
        for i, instance in enumerate(instances):
            name = self.manager.semantic_sensor.get_class_name_for_id(instance.category_id)
            print(f" - Found instance {i} with name {name} and global id {instance.global_id}.")

            if self.show_instances_detected:
                self.show_instance(instance, f"Instance {i} with name {name}")

            # Find a box
            if "box" in name or "tray" in name:
                # Check to see if we can motion plan to box or not
                plan = self.plan_to_instance_for_manipulation(instance, start=start)
                if plan.success:
                    print(f" - Found a reachable box at {instance.get_best_view().get_pose()}.")
                    self.manager.current_receptacle = instance
                    break
                else:
                    self.manager.set_instance_as_unreachable(instance)
                    self.warn(f" - Found a receptacle but could not reach it.")

        # If no receptacle, pick a random point nearby and just wander around
        if self.manager.current_receptacle is None:
            print("None found. Try moving to frontier.")
            # Find a point on the frontier and move there
            res = self.manager.agent.plan_to_frontier(start=start)
            if res.success:
                self.robot.execute_trajectory(
                    [node.state for node in res.trajectory], final_timeout=10.0
                )
                # After moving
                self.update()

                # If we moved to the frontier, then and only then can we clean up the object plans.
                self.warn("Resetting object plans.")
                self.manager.reset_object_plans()
            else:
                self.error("Failed to find a reachable frontier.")
                raise RuntimeError("Failed to find a reachable frontier.")
        else:
            self.cheer(f"Found a receptacle!")
            view = self.manager.current_receptacle.get_best_view()
            image = Image.fromarray(view.get_image())
            image.save("receptacle.png")
            if self.show_map_so_far:
                # This shows us what the robot has found so far
                object_xyz = self.manager.current_receptacle.point_cloud.mean(axis=0).cpu().numpy()
                xyt = self.robot.get_base_pose()
                self.agent.voxel_map.show(
                    orig=object_xyz,
                    xyt=xyt,
                    footprint=self.robot_model.get_footprint(),
                    planner_visuals=False,
                )

    def was_successful(self) -> bool:
        res = self.manager.current_receptacle is not None
        if res:
            self.cheer("Successfully found a receptacle!")
        else:
            self.error("Failed to find a receptacle.")
        return res


class SearchForObjectOnFloorOperation(ManagedOperation):
    """Search for an object on the floor"""

    show_map_so_far: bool = True
    show_instances_detected: bool = False
    plan_for_manipulation: bool = True
    object_class: Optional[str] = None

    def set_target_object_class(self, object_class: str):
        self.warn(f"Overwriting target object class from {self.object_class} to {object_class}.")
        self.object_class = object_class

    def can_start(self) -> bool:
        self.attempt("If receptacle is found, we can start searching for objects.")
        return self.manager.current_receptacle is not None

    def run(self) -> None:
        self.intro("Find a reachable object on the floor.")
        self._successful = False

        # Set the object class if not set
        if self.object_class is None:
            self.object_class = self.manager.target_object

        # Clear the current object
        self.manager.current_object = None

        # Update world map
        # Switch to navigation posture
        self.robot.move_to_nav_posture()
        # Do not update until you are in nav posture
        self.update()

        if self.show_map_so_far:
            # This shows us what the robot has found so far
            xyt = self.robot.get_base_pose()
            self.agent.voxel_map.show(
                orig=np.zeros(3), xyt=xyt, footprint=self.robot_model.get_footprint()
            )

        # Get the current location of the robot
        start = self.robot.get_base_pose()
        if not self.navigation_space.is_valid(start):
            self.error(
                "Robot is in an invalid configuration. It is probably too close to geometry, or localization has failed."
            )
            breakpoint()

        if self.show_instances_detected:
            # Show the last instance image
            import matplotlib

            # TODO: why do we need to configure this every time
            matplotlib.use("TkAgg")
            import matplotlib.pyplot as plt

            plt.imshow(self.manager.voxel_map.observations[0].instance)
            plt.show()

        # Check to see if we have a receptacle in the map
        instances = self.manager.instance_memory.get_instances()

        # Compute scene graph from instance memory so that we can use it
        scene_graph = self.agent.get_scene_graph()

        receptacle_options = []
        print(f"Check explored instances for reachable {self.object_class} instances:")
        for i, instance in enumerate(instances):
            name = self.manager.semantic_sensor.get_class_name_for_id(instance.category_id)
            print(f" - Found instance {i} with name {name} and global id {instance.global_id}.")

            if self.manager.is_instance_unreachable(instance):
                print(" - Instance is unreachable.")
                continue

            if self.show_instances_detected:
                self.show_instance(instance, f"Instance {i} with name {name}")

            if self.object_class in name:
                relations = scene_graph.get_matching_relations(instance.global_id, "floor", "on")
                if len(relations) > 0:
                    # We found a matching relation!
                    print(f" - Found a toy on the floor at {instance.get_best_view().get_pose()}.")

                    # Move to object on floor
                    plan = self.plan_to_instance_for_manipulation(
                        instance, start=start, radius_m=0.5
                    )
                    if plan.success:
                        print(
                            f" - Confirmed toy is reachable with base pose at {plan.trajectory[-1]}."
                        )
                        self.manager.current_object = instance
                        break

        # Check to see if there is a visitable frontier
        if self.manager.current_object is None:
            self.warn(f"No {self.object_class} found. Moving to frontier.")
            # Find a point on the frontier and move there
            res = self.agent.plan_to_frontier(start=start)
            if res.success:
                self.robot.execute_trajectory(
                    [node.state for node in res.trajectory], final_timeout=10.0
                )
            # Update world model once we get to frontier
            self.update()

            # If we moved to the frontier, then and only then can we clean up the object plans.
            self.warn("Resetting object plans.")
            self.manager.reset_object_plans()
        else:
            self.cheer(f"Found object of {self.object_class}!")
            view = self.manager.current_object.get_best_view()
            image = Image.fromarray(view.get_image())
            image.save("object.png")
            if self.show_map_so_far:
                # This shows us what the robot has found so far
                object_xyz = self.manager.current_object.point_cloud.mean(axis=0).cpu().numpy()
                xyt = self.robot.get_base_pose()
                self.agent.voxel_map.show(
                    orig=object_xyz,
                    xyt=xyt,
                    footprint=self.robot_model.get_footprint(),
                    planner_visuals=False,
                )

        # TODO: better behavior
        # If no visitable frontier, pick a random point nearby and just wander around

    def was_successful(self) -> bool:
        return self.manager.current_object is not None and not self.manager.is_instance_unreachable(
            self.manager.current_object
        )


class SearchForObjectOnFloorOperation(ManagedOperation):
    """Search for an object on the floor"""

    # Debugging visualization
    show_map_so_far: bool = False
    show_instances_detected: bool = False

    # Important parameters
    plan_for_manipulation: bool = True
    object_class: Optional[str] = None

    def set_target_object_class(self, object_class: str):
        self.warn(f"Overwriting target object class from {self.object_class} to {object_class}.")
        self.object_class = object_class

    def can_start(self) -> bool:
        self.attempt("If receptacle is found, we can start searching for objects.")
        return self.manager.current_receptacle is not None

    def run(self) -> None:
        self.intro("Find a reachable object on the floor.")
        self._successful = False

        # Set the object class if not set
        if self.object_class is None:
            self.object_class = self.manager.target_object

        # Clear the current object
        self.manager.current_object = None

        # Update world map
        # Switch to navigation posture
        self.robot.move_to_nav_posture()
        # Do not update until you are in nav posture
        self.update()

        if self.show_map_so_far:
            # This shows us what the robot has found so far
            xyt = self.robot.get_base_pose()
            self.agent.voxel_map.show(
                orig=np.zeros(3), xyt=xyt, footprint=self.robot_model.get_footprint()
            )

        # Get the current location of the robot
        start = self.robot.get_base_pose()
        if not self.navigation_space.is_valid(start):
            self.error(
                "Robot is in an invalid configuration. It is probably too close to geometry, or localization has failed."
            )
            breakpoint()

        if self.show_instances_detected:
            # Show the last instance image
            import matplotlib

            # TODO: why do we need to configure this every time
            matplotlib.use("TkAgg")
            import matplotlib.pyplot as plt

            plt.imshow(self.manager.voxel_map.observations[0].instance)
            plt.show()

        # Check to see if we have a receptacle in the map
        instances = self.manager.instance_memory.get_instances()

        # Compute scene graph from instance memory so that we can use it
        scene_graph = self.agent.get_scene_graph()

        receptacle_options = []
        print(f"Check explored instances for reachable {self.object_class} instances:")
        for i, instance in enumerate(instances):
            name = self.manager.semantic_sensor.get_class_name_for_id(instance.category_id)
            print(f" - Found instance {i} with name {name} and global id {instance.global_id}.")

            if self.manager.is_instance_unreachable(instance):
                print(" - Instance is unreachable.")
                continue

            if self.show_instances_detected:
                self.show_instance(instance, f"Instance {i} with name {name}")

            if self.object_class in name:
                relations = scene_graph.get_matching_relations(instance.global_id, "floor", "on")
                if len(relations) > 0:
                    # We found a matching relation!
                    print(f" - Found a toy on the floor at {instance.get_best_view().get_pose()}.")

                    # Move to object on floor
                    plan = self.plan_to_instance_for_manipulation(
                        instance, start=start, radius_m=0.5
                    )
                    if plan.success:
                        print(
                            f" - Confirmed toy is reachable with base pose at {plan.trajectory[-1]}."
                        )
                        self.manager.current_object = instance
                        break

        # Check to see if there is a visitable frontier
        if self.manager.current_object is None:
            self.warn(f"No {self.object_class} found. Moving to frontier.")
            # Find a point on the frontier and move there
            res = self.agent.plan_to_frontier(start=start)
            if res.success:
                self.robot.execute_trajectory(
                    [node.state for node in res.trajectory], final_timeout=10.0
                )
            # Update world model once we get to frontier
            self.update()

            # If we moved to the frontier, then and only then can we clean up the object plans.
            self.warn("Resetting object plans.")
            self.manager.reset_object_plans()
        else:
            self.cheer(f"Found object of {self.object_class}!")
            view = self.manager.current_object.get_best_view()
            image = Image.fromarray(view.get_image())
            image.save("object.png")
            if self.show_map_so_far:
                # This shows us what the robot has found so far
                object_xyz = self.manager.current_object.point_cloud.mean(axis=0).cpu().numpy()
                xyt = self.robot.get_base_pose()
                self.agent.voxel_map.show(
                    orig=object_xyz,
                    xyt=xyt,
                    footprint=self.robot_model.get_footprint(),
                    planner_visuals=False,
                )

        # TODO: better behavior
        # If no visitable frontier, pick a random point nearby and just wander around

    def was_successful(self) -> bool:
        return self.manager.current_object is not None and not self.manager.is_instance_unreachable(
            self.manager.current_object
        )
