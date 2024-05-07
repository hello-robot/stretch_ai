import pprint as pp

import cv2
import numpy as np
import zmq
from scipy.spatial.transform import Rotation

import stretch.demos.dex_teleop.dex_teleop_parameters as dt
import stretch.demos.dex_teleop.goal_from_teleop as gt
import stretch.demos.dex_teleop.webcam_teleop_interface as wt
import stretch.motion.simple_ik as si
from stretch.core import Evaluator
from stretch.demos.dex_teleop.client import RobotClient
from stretch.utils.data_tools.record import FileDataRecorder
from stretch.utils.geometry import get_rotation_from_xyz

use_gripper_center = True


def process_goal_dict(goal_dict, prev_goal_dict=None) -> dict:
    """Process goal dict:
    - fix orientation if necessary
    - calculate relative gripper position and orientation
    - compute quaternion
    - fix offsets if necessary
    """

    # Convert goal dict into a quaternion
    # Start by getting the rotation as a usable object
    r = get_rotation_from_xyz(
        goal_dict["gripper_x_axis"],
        goal_dict["gripper_y_axis"],
        goal_dict["gripper_z_axis"],
    )
    if use_gripper_center:
        # Apply conversion
        # This is a simple frame transformation which should rotate into gripper grasp frame
        delta = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
        r_matrix = r.as_matrix() @ delta
        r = r.from_matrix(r_matrix)
    else:
        goal_dict["gripper_orientation"] = r.as_quat()
        r_matrix = r.as_matrix()

    # Get pose matrix for current frame
    T1 = np.eye(4)
    T1[:3, :3] = r_matrix
    T1[:3, 3] = goal_dict["wrist_position"]

    if use_gripper_center:
        T_wrist_to_grasp = np.eye(4)
        T_wrist_to_grasp[2, 3] = 0
        T_wrist_to_grasp[0, 3] = 0.3
        # T_wrist_to_grasp[1, 3] = 0.3
        T1 = T1 @ T_wrist_to_grasp
        goal_dict["gripper_orientation"] = Rotation.from_matrix(T1[:3, :3]).as_quat()
        goal_dict["gripper_x_axis"] = T1[:3, 0]
        goal_dict["gripper_y_axis"] = T1[:3, 1]
        goal_dict["gripper_z_axis"] = T1[:3, 2]
        goal_dict["wrist_position"] = T1[:3, 3]
        # Note: print debug information; TODO: remove
        # If we need to, we can tune this to center the gripper
        # Charlie had some code which did this in a slightly nicer way, I think
        # print(T1[:3, 3])

    goal_dict["use_gripper_center"] = use_gripper_center
    if prev_goal_dict is not None:

        T0 = np.eye(4)
        r0 = Rotation.from_quat(prev_goal_dict["gripper_orientation"])
        T0[:3, :3] = r0.as_matrix()
        T0[:3, 3] = prev_goal_dict["wrist_position"]

        T = np.linalg.inv(T0) @ T1
        goal_dict["relative_gripper_position"] = T[:3, 3]
        goal_dict["relative_gripper_orientation"] = Rotation.from_matrix(
            T[:3, :3]
        ).as_quat()

    return goal_dict


class DexTeleopLeader(Evaluator):
    """A class for evaluating the DexTeleop system."""

    def __init__(
        self,
        use_fastest_mode: bool = False,
        left_handed: bool = False,
        using_stretch2: bool = False,
        data_dir: str = "./data",
        task_name: str = "task",
        user_name: str = "default_user",
        env_name: str = "default_env",
        force_record: bool = False,
    ):
        super().__init__()

        # TODO: fix these two things
        manipulate_on_ground = False
        slide_lift_range = False

        self.use_fastest_mode = use_fastest_mode
        self.left_handed = left_handed
        self.using_stretch_2 = using_stretch2

        goal_send_context = zmq.Context()
        goal_send_socket = goal_send_context.socket(zmq.PUB)
        goal_send_address = "tcp://*:5555"
        goal_send_socket.setsockopt(zmq.SNDHWM, 1)
        goal_send_socket.setsockopt(zmq.RCVHWM, 1)
        goal_send_socket.bind(goal_send_address)
        self.goal_send_socket = goal_send_socket

        if self.use_fastest_mode:
            if self.using_stretch_2:
                robot_speed = "fastest_stretch_2"
            else:
                robot_speed = "fastest_stretch_3"
        else:
            robot_speed = "slow"
        print("running with robot_speed =", robot_speed)

        lift_middle = dt.get_lift_middle(manipulate_on_ground)
        center_configuration = dt.get_center_configuration(lift_middle)

        if left_handed:
            self.webcam_aruco_detector = wt.WebcamArucoDetector(
                tongs_prefix="left", visualize_detections=False
            )
        else:
            self.webcam_aruco_detector = wt.WebcamArucoDetector(
                tongs_prefix="right", visualize_detections=False
            )

        # Initialize IK
        simple_ik = si.SimpleIK()

        # Define the center position for the wrist that corresponds with
        # the teleop origin.
        self.center_wrist_position = simple_ik.fk_rotary_base(center_configuration)

        self.goal_from_markers = gt.GoalFromMarkers(
            dt.teleop_origin,
            self.center_wrist_position,
            slide_lift_range=slide_lift_range,
        )

        self._force = force_record
        self._recording = False or self._force
        self._need_to_write = False
        self._recorder = FileDataRecorder(data_dir, task_name, user_name, env_name)
        self.prev_goal_dict = None

    def apply(
        self, color_image, depth_image, display_received_images: bool = True
    ) -> dict:
        """Take in image data and other data received by the robot and process it appropriately. Will run the aruco marker detection, predict a goal send that goal to the robot, and save everything to disk for learning."""

        assert (self.camera_info is not None) and (
            self.depth_scale is not None
        ), "ERROR: YoloServoPerception: set_camera_parameters must be called prior to apply. self.camera_info or self.depth_scale is None"

        # Convert depth to meters
        depth_image = depth_image / 1000

        if display_received_images:
            # change depth to be h x w x 3
            depth_image_x3 = np.stack((depth_image,) * 3, axis=-1)
            combined = np.hstack((color_image / 255, depth_image_x3 / 4))
            cv2.imshow("EE RGB/Depth Image", combined)
            # cv2.imshow('Received Depth Image', depth_image)

        # Wait for spacebar to be pressed and start/stop recording
        # Spacebar is 32
        # Escape is 27
        key = cv2.waitKey(1)
        if key == 32:
            self._recording = not self._recording
            self.prev_goal_dict = None
            if self._recording:
                print("[LEADER] Recording started.")
            else:
                print("[LEADER] Recording stopped.")
                self._need_to_write = True
                if self._force:
                    # Try to terminate
                    print("[LEADER] Force recording done. Terminating.")
                    return None
        if key == 27:
            if self._recording:
                self._need_to_write = True
            self._recording = False
            self.set_done()
            print("[LEADER] Recording stopped. Terminating.")

        markers = self.webcam_aruco_detector.process_next_frame()
        goal_dict = self.goal_from_markers.get_goal_dict(markers)

        if goal_dict is not None:
            # Convert goal dict into a quaternion
            process_goal_dict(goal_dict, self.prev_goal_dict)

            if self._recording:
                print("[LEADER] goal_dict =")
                pp.pprint(goal_dict)

            if self._recording and self.prev_goal_dict is not None:
                self._recorder.add(
                    color_image,
                    depth_image,
                    goal_dict["relative_gripper_position"],
                    goal_dict["relative_gripper_orientation"],
                    goal_dict["grip_width"],
                )

            # Send goal_dict to robot
            self.goal_send_socket.send_pyobj(goal_dict)
            self.prev_goal_dict = goal_dict

        if self._need_to_write:
            print("[LEADER] Writing data to disk.")
            self._recorder.write()
            self._need_to_write = False
        return goal_dict

    def __del__(self):
        self.goal_send_socket.close()
        if self._recording or self._need_to_write:
            self._recorder.write()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--robot_ip", type=str, default="192.168.1.15")
    parser.add_argument("-p", "--d405_port", type=int, default=4405)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-u", "--user-name", type=str, default="default_user")
    parser.add_argument("-t", "--task-name", type=str, default="default_task")
    parser.add_argument("-e", "--env-name", type=str, default="default_env")
    parser.add_argument(
        "-r", "--replay", action="store_true", help="Replay a recorded session."
    )
    parser.add_argument(
        "-f", "--force", action="store_true", help="Force data recording."
    )
    parser.add_argument("-d", "--data-dir", type=str, default="./data")
    parser.add_argument(
        "-R",
        "--replay_filename",
        type=str,
        default=None,
        help="The filename of the recorded session to replay, if set..",
    )
    args = parser.parse_args()

    client = RobotClient(
        use_remote_computer=True,
        robot_ip=args.robot_ip,
        d405_port=args.d405_port,
        verbose=args.verbose,
    )

    # Create dex teleop leader - this will detect markers and send off goal dicts to the robot.
    evaluator = DexTeleopLeader(
        data_dir=args.data_dir,
        user_name=args.user_name,
        task_name=args.task_name,
        env_name=args.env_name,
    )
    try:
        client.run(evaluator)
    except KeyboardInterrupt:
        pass
