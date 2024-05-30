#!/usr/bin/env python3

import click
import numpy as np

from stretch.agent.robot_agent import RobotAgent
from stretch.agent.zmq_client import HomeRobotZmqClient
from stretch.core import Parameters, get_parameters
from stretch.core.task import Operation, Task
from stretch.mapping.voxel import SparseVoxelMap, SparseVoxelMapNavigationSpace
from stretch.perception import create_semantic_sensor, get_encoder


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
                breakpoint()

        # Check to see if there is a visitable frontier

        # If no visitable frontier, pick a random point nearby and just wander around

        self._successful = True

    def was_successful(self) -> bool:
        return self._successful and self.manager.current_receptacle is not None


class PreGraspObjectOperation(ManagedOperation):
    """Move the robot to a position looking at the object using the navigation/manipulation camera."""

    def can_start(self):
        return self.manager.current_object is not None

    def run(self):
        print("Moving to a position to grasp the object.")
        self.robot.move_to_manip_posture()
        # This may need to do some more adjustments but this is probab ly ok for now

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


class PickupManager:
    """Simple robot that will look around and pick up different objects"""

    def __init__(self, agent: RobotAgent) -> None:

        # Agent wraps high level functionality
        self.agent = agent

        # Sync these things
        self.robot = agent.robot
        self.voxel_map = agent.voxel_map
        self.navigation_space = agent.space
        self.semantic_sensor = agent.semantic_sensor
        self.parameters = agent.parameters
        self.instance_memory = agent.voxel_map.instances
        assert (
            self.instance_memory is not None
        ), "Make sure instance memory was created! This is configured in parameters file."

        self.current_object = None
        self.current_receptacle = None

    def get_task(self, add_rotate: bool = False) -> Task:
        """Create a task plan with loopbacks and recovery from failure"""

        # Put the robot into navigation mode
        go_to_navigation_mode = GoToNavOperation("go to navigation mode", self)

        if add_rotate:
            # Spin in place to find objects.
            rotate_in_place = RotateInPlaceOperation(
                "Rotate in place", self, parent=go_to_navigation_mode
            )

        # Look for the target receptacle
        search_for_receptacle = SearchForReceptacle(
            "Search for a box",
            self,
            parent=rotate_in_place if add_rotate else go_to_navigation_mode,
        )

        # Try to expand the frontier and find an object; or just wander around for a while.
        search_for_object = SearchForObjectOnFloorOperation(
            "Search for toys on the floor", self, retry_on_failure=True
        )

        # After searching for object, we should go to an instance that we've found. If we cannot do that, keep searching.
        go_to_object = NavigateToObjectOperation(
            "go to object", self, parent=search_for_object, on_cannot_start=search_for_object
        )

        # When about to start, run object detection and try to find the object. If not in front of us, explore again.
        # If we cannot find the object, we should go back to the search_for_object operation.
        # To determine if we can start, we just check to see if there's a detectable object nearby.
        pregrasp_object = PreGraspObjectOperation(
            "prepare to grasp and make sure we can see the object",
            self,
            on_failure=None,
            on_cannot_start=go_to_object,
            retry_on_failure=True,
        )
        # If we cannot start, we should go back to the search_for_object operation.
        # To determine if we can start, we look at the end effector camera and see if there's anything detectable.
        grasp_object = GraspObjectOperation(
            "grasp the object",
            self,
            parent=pregrasp_object,
            on_failure=pregrasp_object,
            on_cannot_start=go_to_object,
        )

        task = Task()
        task.add_operation(go_to_navigation_mode)
        if add_rotate:
            task.add_operation(rotate_in_place)
        task.add_operation(search_for_receptacle)
        task.add_operation(search_for_object)
        task.add_operation(go_to_object)
        task.add_operation(pregrasp_object)
        task.add_operation(grasp_object)

        return task


@click.command()
@click.option("--robot_ip", default="192.168.1.15", help="IP address of the robot")
@click.option("--recv_port", default=4401, help="Port to receive messages from the robot")
@click.option("--send_port", default=4402, help="Port to send messages to the robot")
@click.option("--reset", is_flag=True, help="Reset the robot to origin before starting")
@click.option(
    "--local",
    is_flag=True,
    help="Set if we are executing on the robot and not on a remote computer",
)
@click.option(
    "--parameter_file", default="config/default_planner.yaml", help="Path to parameter file"
)
def main(
    robot_ip: str = "192.168.1.15",
    recv_port: int = 4401,
    send_port: int = 4402,
    local: bool = False,
    parameter_file: str = "config/default_planner.yaml",
    device_id: int = 0,
    verbose: bool = False,
    show_intermediate_maps: bool = False,
    reset: bool = False,
):
    """Set up the robot, create a task plan, and execute it."""
    # Create robot
    parameters = get_parameters(parameter_file)
    robot = HomeRobotZmqClient(
        robot_ip=robot_ip,
        recv_port=recv_port,
        send_port=send_port,
        use_remote_computer=(not local),
        parameters=parameters,
    )
    _, semantic_sensor = create_semantic_sensor(
        device_id=device_id,
        verbose=verbose,
        category_map_file=parameters["open_vocab_category_map_file"],
    )

    # Start moving the robot around
    grasp_client = None

    # Agents wrap the robot high level planning interface for now
    demo = RobotAgent(robot, parameters, semantic_sensor, grasp_client=grasp_client)
    demo.start(visualize_map_at_start=show_intermediate_maps)
    if reset:
        robot.move_to_nav_posture()
        robot.navigate_to([0.0, 0.0, 0.0], blocking=True)

    # After the robot has started...
    try:
        manager = PickupManager(demo)
        task = manager.get_task(add_rotate=False)
    except Exception as e:
        print(f"Error creating task: {e}")
        demo.stop()
        return

    task.execute()

    # At the end, disable everything
    robot.stop()


if __name__ == "__main__":
    main()
