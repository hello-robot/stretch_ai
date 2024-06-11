import time
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation
from termcolor import colored

from stretch.core.task import Operation
from stretch.motion.kinematics import STRETCH_GRASP_OFFSET, HelloStretchIdx
from stretch.utils.geometry import point_global_to_base


class ManagedOperation(Operation):
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
            return

    def was_successful(self) -> bool:
        res = self.manager.current_receptacle is not None
        if res:
            self.cheer("Successfully found a receptacle!")
        else:
            self.error("Failed to find a receptacle.")
        return res


class SearchForObjectOnFloorOperation(ManagedOperation):
    """Search for an object on the floor"""

    show_map_so_far: bool = False
    show_instances_detected: bool = True
    plan_for_manipulation: bool = True

    def can_start(self) -> bool:
        self.attempt("If receptacle is found, we can start searching for objects.")
        return self.manager.current_receptacle is not None

    def run(self) -> None:
        self.intro("Find a reachable object on the floor.")
        self._successful = False

        # Update world map
        # Switch to navigation posture
        self.robot.move_to_nav_posture()
        # Do not update until you are in nav posture
        self.update()

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
                    plan = self.manager.agent.plan_to_instance(
                        instance,
                        start=start,
                        rotation_offset=np.pi / 2 if self.plan_for_manipulation else 0,
                    )
                    if plan.success:
                        print(
                            f" - Confirmed toy is reachable with base pose at {plan.trajectory[-1]}."
                        )
                        self.manager.current_object = instance
                        return

        # Check to see if there is a visitable frontier
        self.warn("Nothing found. Moving to frontier.")
        if self.manager.current_object is None:
            # Find a point on the frontier and move there
            res = self.agent.plan_to_frontier(start=start)
            if res.success:
                self.robot.execute_trajectory(
                    [node.state for node in res.trajectory], final_timeout=10.0
                )
            # Update world model once we get to frontier
            self.update()

        # TODO: better behavior
        # If no visitable frontier, pick a random point nearby and just wander around

    def was_successful(self) -> bool:
        return self.manager.current_object is not None


class PreGraspObjectOperation(ManagedOperation):
    """Move the robot to a position looking at the object using the navigation/manipulation camera."""

    plan = None
    show_object_in_voxel_grid: bool = False
    use_pitch_from_vertical: bool = True
    grasp_distance_threshold: float = 0.8

    def can_start(self):
        self.plan = None
        if self.manager.current_object is None:
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
        plan = self.agent.plan_to_instance(
            self.get_target(),
            start=start,
            rotation_offset=np.pi / 2 if self.for_manipulation else 0,
        )
        if plan.success:
            self.plan = plan
            self.cheer("Found plan to object!")
            return True
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


class GraspObjectOperation(ManagedOperation):
    """Move the robot to grasp, using the end effector camera."""

    use_pitch_from_vertical: bool = True
    lift_distance: float = 0.2
    servo_to_grasp: bool = False
    _success: bool = False

    def can_start(self):
        return self.manager.current_object is not None and self.robot.in_manipulation_mode()

    def visual_servo_to_object(self) -> bool:
        """Use visual servoing to grasp the object."""
        raise NotImplementedError("Visual servoing not implemented yet.")
        return False

    def run(self):
        self.intro("Grasping the object.")
        self._success = False
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
            self._success = self.visual_servo_to_object()
        else:
            target_joint_state, _, _, success, _ = self.robot_model.manip_ik_for_grasp_frame(
                relative_object_xyz, ee_rot, q0=joint_state
            )
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
            # print(f"{self.name}: Lifting the arm up so as not to hit the base.")
            # self.robot.arm_to(target_joint_state_lifted, blocking=True)
            # time.sleep(3.0)
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

    def can_start(self) -> bool:
        self.attempt(
            "will start placing the object if we have object and receptacle, and are close enough to drop."
        )
        if self.manager.current_object is None or self.manager.current_receptacle is None:
            self.error("Object or receptacle not found.")
            return False
        object_xyz = self.manager.current_object.point_cloud.mean(axis=0)
        start = self.robot.get_base_pose()
        dist = np.linalg.norm(object_xyz[:2] - start[:2])
        if dist > self.place_distance_threshold:
            self.error(f"Object is too far away to grasp: {dist}")
            return False
        self.cheer(f"Object is probably close enough to grasp: {dist}")
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

        # End effector position and orientation in global coordinates
        ee_pos, ee_rot = model.manip_fk(joint_state)

        # Switch to place position
        print(" - Move to manip posture")
        self.robot.move_to_manip_posture()

        # Get object xyz coords
        object_xyz = self.manager.current_receptacle.point_cloud.mean(axis=0)
        xyt = self.robot.get_base_pose()

        # Get the center of the object point cloud so that we can place there
        relative_object_xyz = point_global_to_base(object_xyz, xyt)

        # Get max xyz
        max_xyz = self.manager.current_receptacle.point_cloud.max(axis=0)[0]

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
        print("Move to manip posture")
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
