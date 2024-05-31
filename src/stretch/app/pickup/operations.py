import numpy as np

from stretch.core.task import Operation


class ManagedOperation(Operation):
    def __init__(self, name, manager, **kwargs):
        super().__init__(name, **kwargs)
        self.manager = manager
        self.robot = manager.robot
        self.parameters = manager.parameters
        self.navigation_space = manager.navigation_space
        self.agent = manager.agent

    def update(self):
        self.agent.update()


class RotateInPlaceOperation(ManagedOperation):
    """Rotate the robot in place"""

    def can_start(self) -> bool:
        return True

    def run(self) -> None:
        print(
            f"Running {self.name}: rotating for {self.parameters['in_place_rotation_steps']} steps."
        )
        self._successful = False
        self.robot.rotate_in_place(
            steps=self.parameters["in_place_rotation_steps"],
            visualize=False,
        )
        self._successful = True

    def was_successful(self) -> bool:
        return self._successful


class SearchForReceptacle(ManagedOperation):
    """Find a place to put the objects we find on the floor"""

    # For debugging
    show_map_so_far: bool = False
    show_instances_detected: bool = False

    def can_start(self) -> bool:
        return True

    def run(self) -> None:
        """Search for a receptacle on the floor."""

        # Update world map
        self.update()

        print("Searching for a receptacle on the floor.")
        print(f"So far we have found: {len(self.manager.instance_memory)} objects.")

        if self.show_map_so_far:
            # This shows us what the robot has found so far
            self.manager.voxel_map.show(orig=np.zeros(3))

        if self.show_instances_detected:
            # Show the last instance image
            import matplotlib

            # TODO: why do we need to configure this every time
            matplotlib.use("TkAgg")
            import matplotlib.pyplot as plt

            plt.imshow(self.manager.voxel_map.observations[0].instance)
            plt.show()

        # Get the current location of the robot
        start = self.robot.get_base_pose()
        if not self.navigation_space.is_valid(start):
            raise RuntimeError(
                "Robot is in an invalid configuration. It is probably too close to geometry, or localization has failed."
            )

        # Check to see if we have a receptacle in the map
        instances = self.manager.instance_memory.get_instances()
        receptacle_options = []
        print("Check explored instances for reachable receptacles:")
        for i, instance in enumerate(instances):
            name = self.manager.semantic_sensor.get_class_name_for_id(instance.category_id)
            print(f" - Found instance {i} with name {name} and global id {instance.global_id}.")

            if self.show_instances_detected:
                view = instance.get_best_view()
                plt.imshow(view.get_image())
                plt.title(f"Instance {i} with name {name}")
                plt.axis("off")
                plt.show()

            # Find a box
            if "box" in name:
                receptacle_options.append(instance)

                # Check to see if we can motion plan to box or not
                plan = self.manager.agent.plan_to_instance(instance, start=start)
                if plan.success:
                    print(f" - Found a reachable box at {instance.get_best_view().get_pose()}.")
                    self.manager.current_receptacle = instance
                    return

        print("None found. Moving to frontier.")
        # If no receptacle, pick a random point nearby and just wander around
        if self.manager.current_receptacle is None:
            # Find a point on the frontier and move there
            res = self.manager.agent.go_to_frontier(start=start)
            # After moving
            self.update()
            return

    def was_successful(self) -> bool:
        res = self.manager.current_receptacle is not None
        print(f"{self.name}: Successfully found a receptacle!")
        return res


class SearchForObjectOnFloorOperation(ManagedOperation):
    """Search for an object on the floor"""

    show_map_so_far: bool = True
    show_instances_detected: bool = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def can_start(self) -> bool:
        return self.manager.current_receptacle is not None

    def run(self) -> None:
        print("Find a reachable object on the floor.")
        self._successful = False

        # Update world map
        self.agent.update()

        # Get the current location of the robot
        start = self.robot.get_base_pose()
        if not self.navigation_space.is_valid(start):
            raise RuntimeError(
                "Robot is in an invalid configuration. It is probably too close to geometry, or localization has failed."
            )

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
        print("Check explored instances for reachable receptacles:")
        for i, instance in enumerate(instances):
            name = self.manager.semantic_sensor.get_class_name_for_id(instance.category_id)
            print(f" - Found instance {i} with name {name} and global id {instance.global_id}.")
            if self.show_instances_detected:
                view = instance.get_best_view()
                plt.imshow(view.get_image())
                plt.title(f"Instance {i} with name {name}")
                plt.axis("off")
                plt.show()

            if "toy" in name:
                relations = scene_graph.get_matching_relations(instance.global_id, "floor", "on")
                if len(relations) > 0:
                    # We found a matching relation!
                    print(f" - Found a toy on the floor at {instance.get_best_view().get_pose()}.")

                    # Move to object on floor
                    plan = self.manager.agent.plan_to_instance(instance, start=start)
                    if plan.success:
                        print(
                            f" - Confirmed toy is reachable with base pose at {plan.trajectory[-1]}."
                        )
                        self.manager.current_object = instance
                        return

        # Check to see if there is a visitable frontier
        print("Nothing found.   Moving to frontier.")
        if self.manager.current_object is None:
            # Find a point on the frontier and move there
            res = self.agent.go_to_frontier(start=start)

        # TODO: better behavior
        # If no visitable frontier, pick a random point nearby and just wander around

    def was_successful(self) -> bool:
        return self.manager.current_object is not None


class PreGraspObjectOperation(ManagedOperation):
    """Move the robot to a position looking at the object using the navigation/manipulation camera."""

    plan = None
    show_object_in_voxel_grid: bool = False

    def can_start(self):
        self.plan = None
        if self.manager.current_object is None:
            return False

        start = self.robot.get_base_pose()
        if not self.navigation_space.is_valid(start):
            raise RuntimeError(
                "Robot is in an invalid configuration. It is probably too close to geometry, or localization has failed."
            )

        # Motion plan to the object
        plan = self.manager.agent.plan_to_instance(self.manager.current_object, start=start)
        if plan.success:
            self.plan = plan
            return True

    def run(self):

        # Execute the trajectory
        assert (
            self.plan is not None
        ), "Did you make sure that we had a plan? You should call can_start() before run()."
        self.robot.execute_trajectory(self.plan)

        # Orient the robot towards the object and use the end effector camera to pick it up
        xyt = self.plan.trajectory[-1].state
        self.robot.navigate_to(xyt + np.array([0, 0, np.pi / 2]), blocking=True, timeout=10.0)

        print("Moving to a position to grasp the object.")
        self.robot.move_to_manip_posture()

        # Now we should be able to see the object if we orient gripper properly
        # Get the end effector pose
        obs = self.robot.get_observation()
        joint_state = obs.joint
        model = self.robot.get_robot_model()
        ee_pos, ee_rot = model.manip_fk(joint_state)

        # Get the center of the object point cloud so that we can look at it
        object_xyz = self.manager.current_object.point_cloud.mean(axis=0)
        if self.show_object_in_voxel_grid:
            self.agent.voxel_map.show(orig=object_xyz.cpu().numpy())
        from stretch.utils.geometry import point_global_to_base, xyt_global_to_base

        xyt = self.robot.get_base_pose()
        relative_xyz = point_global_to_base(object_xyz, xyt)

        # Compute the angles necessary

        breakpoint()

    def was_successful(self):
        return self.robot.in_manipulation_mode()


class NavigateToObjectOperation(ManagedOperation):
    def can_start(self):
        return self.manager.current_object is not None

    def run(self):
        print("Navigating to the object.")
        self.robot.move_to_nav_posture()

        # Now find the object instance we got from the map

    def was_successful(self):
        """This will be successful if we got within a reasonable distance of the target object."""
        return self.robot.in_navigation_mode()


class GraspObjectOperation(ManagedOperation):
    """Move the robot to grasp, using the end effector camera."""

    def can_start(self):
        return self.manager.current_object is not None

    def run(self):
        breakpoint()

    def was_successful(self):
        breakpoint()


class GoToNavOperation(ManagedOperation):
    """Put the robot into navigation mode"""

    def can_start(self) -> bool:
        return True

    def run(self) -> None:
        print("Switching to navigation mode.")
        self.robot.move_to_nav_posture()

    def was_successful(self) -> bool:
        return self.robot.in_navigation_mode()
