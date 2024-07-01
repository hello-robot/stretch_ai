import time
import timeit
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from termcolor import colored

from stretch.core.interfaces import Observations
from stretch.core.task import Operation
from stretch.mapping.instance import Instance
from stretch.motion.kinematics import HelloStretchIdx
from stretch.utils.geometry import point_global_to_base


class ManagedOperation(Operation):
    """Placeholder node for an example in a task plan. Contains some functions to make it easier to print out different messages with color for interpretability, and also provides some utilities for making the robot do different tasks."""

    def __init__(self, name, manager, **kwargs):
        super().__init__(name, **kwargs)
        self.manager = manager
        self.robot = manager.robot
        self.parameters = manager.parameters
        self.navigation_space = manager.navigation_space
        self.agent = manager.agent
        self.robot_model = self.robot.get_robot_model()

    def update(self):
        print(colored("================ Updating the world model ==================", "blue"))
        self.agent.update()

    def attempt(self, message: str):
        print(colored(f"Trying {self.name}:", "blue"), message)

    def intro(self, message: str):
        print(colored(f"Running {self.name}:", "green"), message)

    def warn(self, message: str):
        print(colored(f"Warning in {self.name}: {message}", "yellow"))

    def error(self, message: str):
        print(colored(f"Error in {self.name}: {message}", "red"))

    def cheer(self, message: str):
        """An upbeat message!"""
        print(colored(f"!!! {self.name} !!!: {message}", "green"))

    def plan_to_instance_for_manipulation(self, instance, start, radius_m: float = 0.45):
        """Manipulation planning wrapper. Plan to instance with a radius around it, ensuring a base location can be found in explored space."""
        return self.agent.plan_to_instance(
            instance, start=start, rotation_offset=np.pi / 2, radius_m=radius_m, max_tries=100
        )

    def show_instance_segmentation_image(self):
        # Show the last instance image
        import matplotlib

        # TODO: why do we need to configure this every time
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt

        plt.imshow(self.manager.voxel_map.observations[0].instance)
        plt.show()

    def show_instance(self, instance: Instance, title: Optional[str] = None):
        """Show the instance in the voxel grid."""
        import matplotlib

        matplotlib.use("TkAgg")

        import matplotlib.pyplot as plt

        view = instance.get_best_view()
        plt.imshow(view.get_image())
        if title is not None:
            plt.title(title)
        plt.axis("off")
        plt.show()


class RotateInPlaceOperation(ManagedOperation):
    """Rotate the robot in place"""

    def can_start(self) -> bool:
        self.attempt(f"rotating for {self.parameters['in_place_rotation_steps']} steps.")
        return True

    def run(self) -> None:
        self.intro("rotating for {self.parameters['in_place_rotation_steps']} steps.")
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
        receptacle_options = []
        print("Check explored instances for reachable receptacles:")
        for i, instance in enumerate(instances):
            name = self.manager.semantic_sensor.get_class_name_for_id(instance.category_id)
            print(f" - Found instance {i} with name {name} and global id {instance.global_id}.")

            if self.show_instances_detected:
                self.show_instance(instance, f"Instance {i} with name {name}")

            # Find a box
            if "box" in name or "tray" in name:
                receptacle_options.append(instance)

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
            else:
                self.error("Failed to find a reachable frontier.")
                raise RuntimeError("Failed to find a reachable frontier.")
            # After moving
            self.update()
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


class PreGraspObjectOperation(ManagedOperation):
    """Move the robot to a position looking at the object using the navigation/manipulation camera."""

    plan = None
    show_object_in_voxel_grid: bool = False
    use_pitch_from_vertical: bool = True
    grasp_distance_threshold: float = 0.8

    def can_start(self):
        """Can only move to an object if it's been picked out and is reachable."""

        self.plan = None
        if self.manager.current_object is None:
            return False
        elif self.manager.is_instance_unreachable(self.manager.current_object):
            return False

        start = self.robot.get_base_pose()
        if not self.navigation_space.is_valid(start):
            self.error(
                f"{self.name}: [ERROR]: Robot is in an invalid configuration. It is probably too close to geometry, or localization has failed."
            )
            breakpoint()

        # Get the center of the object point cloud so that we can look at it
        object_xyz = self.manager.current_object.point_cloud.mean(axis=0)
        dist = np.linalg.norm(object_xyz[:2] - start[:2])
        if dist > self.grasp_distance_threshold:
            self.error(f"Object is too far away to grasp: {dist}")
            return False
        self.cheer(f"{self.name}: Object is probably close enough to grasp: {dist}")
        return True

    def run(self):

        self.intro("Moving to a position to grasp the object.")
        self.robot.move_to_manip_posture()

        # Now we should be able to see the object if we orient gripper properly
        # Get the end effector pose
        obs = self.robot.get_observation()
        joint_state = obs.joint
        model = self.robot.get_robot_model()

        # Note that these are in the robot's current coordinate frame; they're not global coordinates, so this is ok to use to compute motions.
        ee_pos, ee_rot = model.manip_fk(joint_state)

        # Get the center of the object point cloud so that we can look at it
        object_xyz = self.manager.current_object.point_cloud.mean(axis=0)
        xyt = self.robot.get_base_pose()
        if self.show_object_in_voxel_grid:
            # Show where the object is together with the robot base
            self.agent.voxel_map.show(
                orig=object_xyz.cpu().numpy(), xyt=xyt, footprint=self.robot_model.get_footprint()
            )
        relative_object_xyz = point_global_to_base(object_xyz, xyt)

        # Compute the angles necessary
        if self.use_pitch_from_vertical:
            # dy = relative_gripper_xyz[1] - relative_object_xyz[1]
            dy = np.abs(ee_pos[1] - relative_object_xyz[1])
            dz = np.abs(ee_pos[2] - relative_object_xyz[2])
            pitch_from_vertical = np.arctan2(dy, dz)
            # current_ee_pitch = joint_state[HelloStretchIdx.WRIST_PITCH]
        else:
            pitch_from_vertical = 0.0

        # Joint state goal
        joint_state[HelloStretchIdx.WRIST_PITCH] = -np.pi / 2 + pitch_from_vertical

        # Strip out fields from the full robot state to only get the 6dof manipulator state
        # TODO: we should probably handle this in the zmq wrapper.
        # arm_cmd = self.robot_model.config_to_manip_command(joint_state)
        self.robot.arm_to(joint_state, blocking=True)

        # It does not take long to execute these commands
        time.sleep(2.0)

    def was_successful(self):
        return self.robot.in_manipulation_mode()


class NavigateToObjectOperation(ManagedOperation):

    plan = None
    for_manipulation: bool = True
    be_precise: bool = False

    def __init__(self, *args, to_receptacle=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.to_receptacle = to_receptacle

    def get_target(self):
        if self.to_receptacle:
            return self.manager.current_receptacle
        else:
            return self.manager.current_object

    def can_start(self):
        print(
            f"{self.name}: check to see if object is reachable (receptacle={self.to_receptacle})."
        )
        self.plan = None
        if self.get_target() is None:
            self.error("no target!")
            return False

        start = self.robot.get_base_pose()
        if not self.navigation_space.is_valid(start):
            self.error(
                "Robot is in an invalid configuration. It is probably too close to geometry, or localization has failed."
            )
            breakpoint()

        # Motion plan to the object
        plan = self.plan_to_instance_for_manipulation(self.get_target(), start=start)
        if plan.success:
            self.plan = plan
            self.cheer("Found plan to object!")
            return True
        else:
            self.manager.set_instance_as_unreachable(self.get_target())
        self.error("Planning failed!")
        return False

    def run(self):
        self.intro("executing motion plan to the object.")
        self.robot.move_to_nav_posture()

        # Execute the trajectory
        assert (
            self.plan is not None
        ), "Did you make sure that we had a plan? You should call can_start() before run()."
        self.robot.execute_trajectory(self.plan, final_timeout=10.0)

        # Orient the robot towards the object and use the end effector camera to pick it up
        xyt = self.plan.trajectory[-1].state
        # self.robot.navigate_to(xyt + np.array([0, 0, np.pi / 2]), blocking=True, timeout=30.0)
        if self.be_precise:
            self.warn("Moving again to make sure we're close enough to the goal.")
            self.robot.navigate_to(xyt, blocking=True, timeout=30.0)

    def was_successful(self):
        """This will be successful if we got within a reasonable distance of the target object."""
        return True  # self.robot.in_navigation_mode()


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
            time.sleep(5.0)
        self.robot.arm_to([0.0, 0.4, 0.05, np.pi, -np.pi / 4, 0], blocking=True)
        time.sleep(8.0)
        xyt = self.robot.get_base_pose()
        # Now update the world
        self.update()
        # Delete observations near us, since they contain the arm!!
        self.agent.voxel_map.delete_obstacles(point=xyt[:2], radius=0.7)

        # Notify and move the arm back to normal. Showing the map is optional.
        print(f"So far we have found: {len(self.manager.instance_memory)} objects.")
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

            plt.imshow(self.manager.voxel_map.observations[-1].instance)
            plt.show()

        # Describe the scene the robot is operating in
        scene_graph = self.agent.get_scene_graph()

        # Get the current location of the robot
        start = self.robot.get_base_pose()
        instances = self.manager.instance_memory.get_instances()
        receptacle_options = []
        object_options = []
        dist_to_object = float("inf")

        print("Check explored instances for reachable receptacles:")
        for i, instance in enumerate(instances):
            name = self.manager.semantic_sensor.get_class_name_for_id(instance.category_id)
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
                    self.manager.current_receptacle = instance
                else:
                    self.warn(f" - Found a receptacle but could not reach it.")
            elif self.manager.target_object in name:
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
                        self.manager.current_object = instance
                        dist_to_object = dist
                else:
                    self.warn(f" - Found an object of class {name} but could not reach it.")

    def was_successful(self):
        """We're just taking an image so this is always going to be a success"""
        return self.manager.current_object is not None


class GraspObjectOperation(ManagedOperation):
    """Move the robot to grasp, using the end effector camera."""

    use_pitch_from_vertical: bool = True
    lift_distance: float = 0.2
    servo_to_grasp: bool = False
    _success: bool = False
    show_object_to_grasp: bool = False
    show_servo_gui: bool = True

    # Thresholds for centering on object
    align_x_threshold: int = 10
    align_y_threshold: int = 5

    # Visual servoing config
    min_points_to_approach: int = 100
    lift_arm_ratio: float = 0.1
    base_x_step: float = 0.03
    wrist_pitch_step: float = 0.025
    median_distance_when_grasping: float = 0.175
    percentage_of_image_when_grasping: float = 0.2

    # Timing issues
    expected_network_delay = 0.75
    open_loop: bool = False

    def can_start(self):
        """Grasping can start if we have a target object picked out, and are moving to its instance, and if the robot is ready to begin manipulation."""
        return self.manager.current_object is not None and self.robot.in_manipulation_mode()

    def get_target_mask(
        self, servo: Observations, instance: Instance, prev_mask: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """Get target mask to move to. If we do not provide the mask from the previous step, we will simply find the mask with the most points of the correct class. Otherwise, we will try to find the mask that most overlaps with the previous mask. There are two options here: one where we simply find the mask with the most points, and another where we try to find the mask that most overlaps with the previous mask. This is in case we are losing track of particular objects and getting classes mixed up.

        Args:
            servo (Observations): Servo observation
            instance (Instance): Instance we are trying to grasp
            prev_mask (Optional[np.ndarray], optional): Mask from the previous step. Defaults to None.

        Returns:
            Optional[np.ndarray]: Target mask to move to
        """
        # Find the best masks
        class_mask = servo.semantic == instance.get_category_id()
        instance_mask = servo.instance
        target_mask = None
        target_mask_pts = float("-inf")
        maximum_overlap_mask = None
        maximum_overlap_pts = float("-inf")
        H, W = class_mask.shape
        for iid in np.unique(instance_mask):
            current_instance_mask = instance_mask == iid

            # If we are centered on the mask and it's the right class, just go for it
            if class_mask[H // 2, W // 2] > 0 and current_instance_mask[H // 2, W // 2] > 0:
                # This is the correct one - it's centered and the right class. Just go there.
                print("!!! CENTERED ON THE RIGHT OBJECT !!!")
                return current_instance_mask

            # Option 2 - try to find the map that most overlapped with what we were just trying to grasp
            # This is in case we are losing track of particular objects and getting classes mixed up
            if prev_mask is not None:
                # Find the mask with the most points
                mask = np.bitwise_and(current_instance_mask, prev_mask)
                num_pts = sum(mask.flatten())

                if num_pts > maximum_overlap_pts:
                    maximum_overlap_pts = num_pts
                    maximum_overlap_mask = mask

            # Simply find the mask with the most points
            mask = np.bitwise_and(instance_mask == iid, class_mask)
            num_pts = sum(mask.flatten())
            if num_pts > target_mask_pts:
                target_mask = mask
                target_mask_pts = num_pts

        if maximum_overlap_pts > self.min_points_to_approach:
            return maximum_overlap_mask
        elif target_mask is not None:
            return target_mask
        else:
            return prev_mask

    def _grasp(self) -> bool:
        """Helper function to close gripper around object."""
        self.cheer("Grasping object!")
        self.robot.close_gripper(blocking=True)
        time.sleep(2.0)

        # Get a joint state for the object
        joint_state = self.robot.get_joint_state()

        # Lifted joint state
        lifted_joint_state = joint_state.copy()
        lifted_joint_state[HelloStretchIdx.LIFT] += 0.2
        self.robot.arm_to(lifted_joint_state, blocking=True)
        time.sleep(2.0)
        return True

    def visual_servo_to_object(self, instance: Instance, max_duration: float = 120.0) -> bool:
        """Use visual servoing to grasp the object."""

        self.intro(f"Visual servoing to grasp object {instance.global_id} {instance.category_id=}.")
        if self.show_servo_gui:
            self.warn("If you want to stop the visual servoing with the GUI up, press 'q'.")

        t0 = timeit.default_timer()
        aligned_once = False
        prev_target_mask = None
        success = False

        # Get joint state observation
        joint_state = self.robot.get_joint_state()
        base_x = joint_state[HelloStretchIdx.BASE_X]
        wrist_pitch = joint_state[HelloStretchIdx.WRIST_PITCH]
        arm = joint_state[HelloStretchIdx.ARM]
        lift = joint_state[HelloStretchIdx.LIFT]

        while timeit.default_timer() - t0 < max_duration:

            # Get servo observation
            servo = self.robot.get_servo_observation()
            joint_state = self.robot.get_joint_state()

            if not self.open_loop:
                # Now compute what to do
                base_x = joint_state[HelloStretchIdx.BASE_X]
                wrist_pitch = joint_state[HelloStretchIdx.WRIST_PITCH]
                arm = joint_state[HelloStretchIdx.ARM]
                lift = joint_state[HelloStretchIdx.LIFT]

            # Run semantic segmentation on it
            servo = self.agent.semantic_sensor.predict(servo, ee=True)
            target_mask = self.get_target_mask(servo, instance, prev_mask=prev_target_mask)
            target_mask_pts = sum(target_mask.flatten())

            # Optionally display which object we are servoing to
            if self.show_servo_gui:
                servo_ee_rgb = cv2.cvtColor(servo.ee_rgb, cv2.COLOR_RGB2BGR)
                mask = target_mask.astype(np.uint8) * 255
                mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                mask[:, :, 0] = 0
                servo_ee_rgb = cv2.addWeighted(servo_ee_rgb, 0.5, mask, 0.5, 0, servo_ee_rgb)
                cv2.imshow("servo_ee_rgb", servo_ee_rgb)
                cv2.waitKey(1)
                res = cv2.waitKey(1) & 0xFF  # 0xFF is a mask to get the last 8 bits
                if res == ord("q"):
                    break

            center_depth = (
                servo.ee_depth[servo.ee_depth.shape[0] // 2, servo.ee_depth.shape[1] // 2] / 1000
            )
            if target_mask is not None and target_mask_pts > self.min_points_to_approach:
                object_depth = servo.ee_depth[target_mask]
                median_object_depth = np.median(servo.ee_depth[target_mask]) / 1000
            else:
                print("detected classes:", np.unique(servo.semantic))
                if center_depth < self.median_distance_when_grasping:
                    success = self._grasp()
                continue

            # Compute the center of the mask in image coords
            mask = target_mask
            mask_pts = np.argwhere(mask)
            mask_center = mask_pts.mean(axis=0)
            center_x, center_y = target_mask.shape[1] / 2, target_mask.shape[0] / 2
            dx, dy = mask_center[1] - center_x, mask_center[0] - center_y

            # Is the center of the image part of the target mask or not?
            center_in_mask = target_mask[int(center_y), int(center_x)] > 0

            # Since we were able to detect it, copy over the target mask
            prev_target_mask = target_mask

            print()
            print("----- STEP VISUAL SERVOING -----")
            print("cur x =", base_x)
            print(" lift =", lift)
            print("  arm =", arm)
            print("pitch =", wrist_pitch)
            print(f"base_x={base_x}, wrist_pitch={wrist_pitch}, dx={dx}, dy={dy}")
            print(f"Median distance to object is {median_object_depth}.")
            print(f"Center distance to object is {center_depth}.")
            print("Center in mask?", center_in_mask)
            percentage_of_image = mask_pts.shape[0] / (target_mask.shape[0] * target_mask.shape[1])
            print(f"Percentage of image with object is {percentage_of_image}.")
            if center_in_mask and (
                center_depth < self.median_distance_when_grasping
                or median_object_depth < self.median_distance_when_grasping
            ):
                "If there's any chance the object is close enough, we should just try to grasp it." ""
                success = self._grasp()
                break
            elif np.abs(dx) < self.align_x_threshold and np.abs(dy) < self.align_y_threshold:
                # First, check to see if we are close enough to grasp
                if center_depth < self.median_distance_when_grasping:
                    success = self._grasp()
                    break
                # If we are aligned, step the whole thing closer by some amount
                # This is based on the pitch - basically
                aligned_once = True
                arm_component = np.cos(wrist_pitch) * self.lift_arm_ratio
                lift_component = np.sin(wrist_pitch) * self.lift_arm_ratio
                arm += arm_component
                lift += lift_component
            else:
                # Add these to do some really hacky proportionate control
                px = max(0.25, np.abs(2 * dx / target_mask.shape[1]))
                py = max(0.25, np.abs(2 * dy / target_mask.shape[0]))

                # Move the base and modify the wrist pitch
                print(f"dx={dx}, dy={dy}, px={px}, py={py}")
                if dx > self.align_x_threshold:
                    # Move in x - this means translate the base
                    base_x += -self.base_x_step * px
                elif dx < -1 * self.align_x_threshold:
                    base_x += self.base_x_step * px
                if dy > self.align_y_threshold:
                    # Move in y - this means translate the base
                    wrist_pitch += -self.wrist_pitch_step * py
                elif dy < -1 * self.align_y_threshold:
                    wrist_pitch += self.wrist_pitch_step * py

                # Force to reacquire the target mask if we moved the camera too much
                prev_target_mask = None

            print("tgt x =", base_x)
            print(" lift =", lift)
            print("  arm =", arm)
            print("pitch =", wrist_pitch)

            # breakpoint()
            self.robot.arm_to([base_x, lift, arm, 0, wrist_pitch, 0], blocking=True)

            joint_state[HelloStretchIdx.BASE_X] = base_x
            joint_state[HelloStretchIdx.LIFT] = lift
            joint_state[HelloStretchIdx.ARM] = arm
            joint_state[HelloStretchIdx.WRIST_PITCH] = wrist_pitch

            time.sleep(self.expected_network_delay)

            # Optionally show depth and xyz
            rgb, depth = servo.ee_rgb, servo.ee_depth

            # Depth should be metric depth from camera
            # print("TODO: compute motion based on these points")

            if sum(mask.flatten()) > self.min_points_to_approach:
                # Move towards the points
                pass

        return success

    def run(self):
        self.intro("Grasping the object.")
        self._success = False
        if self.show_object_to_grasp:
            self.show_instance(self.manager.current_object)

        # Now we should be able to see the object if we orient gripper properly
        # Get the end effector pose
        obs = self.robot.get_observation()
        joint_state = obs.joint
        model = self.robot.get_robot_model()

        if joint_state[HelloStretchIdx.GRIPPER] < 0.0:
            self.robot.open_gripper(blocking=True)

        # Get the current base pose of the robot
        xyt = self.robot.get_base_pose()

        # Note that these are in the robot's current coordinate frame; they're not global coordinates, so this is ok to use to compute motions.
        ee_pos, ee_rot = model.manip_fk(joint_state)
        object_xyz = self.manager.current_object.point_cloud.mean(axis=0)
        relative_object_xyz = point_global_to_base(object_xyz, xyt)

        # Compute the angles necessary
        if self.use_pitch_from_vertical:
            # dy = relative_gripper_xyz[1] - relative_object_xyz[1]
            dy = np.abs(ee_pos[1] - relative_object_xyz[1])
            dz = np.abs(ee_pos[2] - relative_object_xyz[2])
            pitch_from_vertical = np.arctan2(dy, dz)
            # current_ee_pitch = joint_state[HelloStretchIdx.WRIST_PITCH]
        else:
            pitch_from_vertical = 0.0

        # Joint state goal
        joint_state[HelloStretchIdx.WRIST_PITCH] = -np.pi / 2 + pitch_from_vertical

        # Strip out fields from the full robot state to only get the 6dof manipulator state
        # TODO: we should probably handle this in the zmq wrapper.
        # arm_cmd = self.robot_model.config_to_manip_command(joint_state)
        self.robot.arm_to(joint_state, blocking=True)

        if self.servo_to_grasp:
            self._success = self.visual_servo_to_object(self.manager.current_object)
        else:
            target_joint_state, _, _, success, _ = self.robot_model.manip_ik_for_grasp_frame(
                relative_object_xyz, ee_rot, q0=joint_state
            )
            target_joint_state[HelloStretchIdx.BASE_X] -= 0.04
            if not success:
                print("Failed to find a valid IK solution.")
                self._success = False
                return
            elif (
                target_joint_state[HelloStretchIdx.ARM] < 0
                or target_joint_state[HelloStretchIdx.LIFT] < 0
            ):
                print(
                    f"{self.name}: Target joint state is invalid: {target_joint_state}. Positions for arm and lift must be positive."
                )
                self._success = False
                return

            # Lift the arm up a bit
            target_joint_state_lifted = target_joint_state.copy()
            target_joint_state_lifted[HelloStretchIdx.LIFT] += self.lift_distance

            # Move to the target joint state
            print(f"{self.name}: Moving to grasp position.")
            self.robot.arm_to(target_joint_state, blocking=True)
            time.sleep(3.0)
            print(f"{self.name}: Closing the gripper.")
            self.robot.close_gripper(blocking=True)
            time.sleep(2.0)
            print(f"{self.name}: Lifting the arm up so as not to hit the base.")
            self.robot.arm_to(target_joint_state_lifted, blocking=False)
            time.sleep(2.0)
            print(f"{self.name}: Return arm to initial configuration.")
            self.robot.arm_to(joint_state, blocking=True)
            time.sleep(3.0)
            print(f"{self.name}: Done.")
            self._success = True

    def was_successful(self):
        """Return true if successful"""
        return self._success


class GoToNavOperation(ManagedOperation):
    """Put the robot into navigation mode"""

    def can_start(self) -> bool:
        self.attempt("will switch to navigation mode.")
        return True

    def run(self) -> None:
        self.intro("Switching to navigation mode.")
        self.robot.move_to_nav_posture()

    def was_successful(self) -> bool:
        return self.robot.in_navigation_mode()


class PlaceObjectOperation(ManagedOperation):
    """Place an object on top of the target receptacle, by just using the arm for now."""

    place_distance_threshold: float = 0.8
    lift_distance: float = 0.2
    place_height_margin: float = 0.1
    show_place_in_voxel_grid: bool = False
    place_step_size: float = 0.15
    use_pitch_from_vertical: bool = True

    def get_target(self):
        return self.manager.current_receptacle

    def get_target_center(self):
        return self.get_target().point_cloud.mean(axis=0)

    def sample_placement_position(self, xyt) -> np.ndarray:
        """Sample a placement position for the object on the receptacle."""
        if self.get_target() is None:
            raise RuntimeError("no target set")
        target = self.get_target()
        center_xyz = self.get_target_center()
        print(" - Placing object on receptacle at", center_xyz)

        # Get the point cloud of the object and find distances to robot
        distances = (target.point_cloud[:, :2] - xyt[:2]).norm(dim=1)
        # Choose closest point to xyt
        idx = distances.argmin()
        # Get the point
        point = target.point_cloud[idx].cpu().numpy()
        print(" - Closest point to robot is", point)
        print(" - Distance to robot is", distances[idx])
        # Compute distance to the center of the object
        distance = np.linalg.norm(point[:2] - center_xyz[:2].cpu().numpy())
        # Take a step towards the center of the object
        dxyz = (center_xyz - point).cpu().numpy()
        point[:2] = point[:2] + (
            dxyz[:2] / np.linalg.norm(dxyz[:2]) * min(distance, self.place_step_size)
        )
        print(" - After taking a step towards the center of the object, we are at", point)
        return point

    def can_start(self) -> bool:
        self.attempt(
            "will start placing the object if we have object and receptacle, and are close enough to drop."
        )
        if self.manager.current_object is None or self.manager.current_receptacle is None:
            self.error("Object or receptacle not found.")
            return False
        object_xyz = self.get_target_center()
        start = self.robot.get_base_pose()
        dist = np.linalg.norm(object_xyz[:2] - start[:2])
        if dist > self.place_distance_threshold:
            self.error(f"Object is too far away to grasp: {dist}")
            return False
        self.cheer(f"Object is probably close enough to place upon: {dist}")
        return True

    def _get_place_joint_state(
        self, pos: np.ndarray, quat: np.ndarray, joint_state: Optional[np.ndarray] = None
    ):
        """Use inverse kinematics to compute joint position for (pos, quat) in base frame.

        Args:
            pos: 3D position of the target in the base frame
            quat: 4D quaternion of the target in the base frame
            joint_state: current joint state of the robot (optional) for inverse kinematics
        """
        if joint_state is None:
            joint_state = self.robot.get_observation().joint

        target_joint_state, _, _, success, _ = self.robot_model.manip_ik_for_grasp_frame(
            pos, quat, q0=joint_state
        )

        return target_joint_state, success

    def run(self) -> None:
        self.intro("Placing the object on the receptacle.")
        self._successful = False

        # Get initial (carry) joint posture
        obs = self.robot.get_observation()
        joint_state = obs.joint
        model = self.robot.get_robot_model()

        # Switch to place position
        print(" - Move to manip posture")
        self.robot.move_to_manip_posture()

        # Get object xyz coords
        xyt = self.robot.get_base_pose()
        placement_xyz = self.sample_placement_position(xyt)
        print(" - Place object at", placement_xyz)

        # Get the center of the object point cloud so that we can place there
        relative_object_xyz = point_global_to_base(placement_xyz, xyt)

        # Compute the angles necessary
        if self.use_pitch_from_vertical:
            ee_pos, ee_rot = model.manip_fk(joint_state)
            # dy = relative_gripper_xyz[1] - relative_object_xyz[1]
            dy = np.abs(ee_pos[1] - relative_object_xyz[1])
            dz = np.abs(ee_pos[2] - relative_object_xyz[2])
            pitch_from_vertical = np.arctan2(dy, dz)
            # current_ee_pitch = joint_state[HelloStretchIdx.WRIST_PITCH]
        else:
            pitch_from_vertical = 0.0

        # Joint compute a joitn state goal and associated ee pos/rot
        joint_state[HelloStretchIdx.WRIST_PITCH] = -np.pi / 2 + pitch_from_vertical
        self.robot.arm_to(joint_state)
        ee_pos, ee_rot = model.manip_fk(joint_state)

        # Get max xyz
        max_xyz = self.get_target().point_cloud.max(axis=0)[0]

        # Placement is at xy = object_xyz[:2], z = max_xyz[2] + margin
        place_xyz = np.array(
            [relative_object_xyz[0], relative_object_xyz[1], max_xyz[2] + self.place_height_margin]
        )

        if self.show_place_in_voxel_grid:
            self.agent.voxel_map.show(
                orig=place_xyz, xyt=xyt, footprint=self.robot_model.get_footprint()
            )

        target_joint_state, success = self._get_place_joint_state(
            pos=place_xyz, quat=ee_rot, joint_state=joint_state
        )
        self.attempt(f"Trying to place the object on the receptacle at {place_xyz}.")
        if not success:
            self.error("Could not place object!")
            return

        # Move to the target joint state
        self.robot.arm_to(target_joint_state, blocking=True)
        time.sleep(5.0)

        # Open the gripper
        self.robot.open_gripper(blocking=True)
        time.sleep(2.0)

        # Move directly up
        target_joint_state_lifted = target_joint_state.copy()
        target_joint_state_lifted[HelloStretchIdx.LIFT] += self.lift_distance
        self.robot.arm_to(target_joint_state_lifted, blocking=True)
        time.sleep(5.0)

        # Return arm to initial configuration and switch to nav posture
        self.robot.move_to_nav_posture()
        time.sleep(2.0)
        self._successful = True

        self.cheer("We believe we successfully placed the object.")

    def was_successful(self):
        self.error("Success detection not implemented.")
        return self._successful
