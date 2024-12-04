# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# (c) 2024 Hello Robot by Atharva Pusalkar
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import datetime
from pathlib import Path

import click
import numpy as np

# Mapping and perception
from stretch.agent.robot_agent import RobotAgent
from stretch.agent.zmq_client import HomeRobotZmqClient
from stretch.app.lfd.ros2_lfd_leader import ROS2LfdLeader
from stretch.core import get_parameters
from stretch.llms.gemma_client import Gemma2bClient
from stretch.llms.prompts.object_manip_nav_prompt import ObjectManipNavPromptBuilder

# from stretch.perception import create_semantic_sensor


@click.command()
@click.option("--local", is_flag=True, help="Run code locally on the robot.")
@click.option("--recv_port", default=4401, help="Port to receive observations on")
@click.option("--send_port", default=4402, help="Port to send actions to on the robot")
@click.option("--robot_ip", default="")
@click.option("--output-filename", default="stretch_output", type=str)
@click.option("--explore-iter", default=0)
@click.option("--spin", default=False, is_flag=True)
@click.option("--reset", is_flag=True)
@click.option(
    "--input_file", default="", type=str, help="Path to input file used instead of robot data"
)
@click.option(
    "--write-instance-images",
    default=False,
    is_flag=True,
    help="write out images of every object we found",
)
@click.option("--parameter-file", default="default_planner.yaml")
@click.option("--reset", is_flag=True, help="Reset the robot to origin before starting")
@click.option("--frame", default=-1, help="Final frame to read from input file")
@click.option("--text", default="", help="Text to encode")
@click.option("-y", "--yes", is_flag=True, help="Skip confirmation")
@click.option(
    "--all-matches",
    is_flag=True,
    help="Find all objects with a similarity to the query above some threshold",
)
# This threshold seems to work ok for Siglip - will not work for e.g. CLIP
@click.option("--threshold", default=0.05, help="Threshold for similarity when using --all-matches")
@click.option(
    "--stationary",
    is_flag=True,
    help="Don't move the robot to the instance, if using real robot instead of offline data",
)
@click.option(
    "--enable-realtime-updates",
    "--enable_realtime_updates",
    is_flag=True,
    help="Enable real-time updates so that the robot will dynamically update the map as it moves",
)
def main(
    device_id: int = 0,
    verbose: bool = True,
    parameter_file: str = "config/default_planner.yaml",
    local: bool = False,
    recv_port: int = 4401,
    send_port: int = 4402,
    robot_ip: str = "",
    reset: bool = False,
    explore_iter: int = 0,
    output_filename: str = "stretch_output",
    spin: bool = False,
    write_instance_images: bool = False,
    input_file: str = "",
    frame: int = -1,
    text: str = "",
    yes: bool = False,
    stationary: bool = False,
    all_matches: bool = False,
    threshold: float = 0.5,
    enable_realtime_updates: bool = False,
):

    print("- Load parameters")
    parameters = get_parameters(parameter_file)
    # semantic_sensor = create_semantic_sensor(
    #     device_id=device_id,
    #     verbose=verbose,
    #     category_map_file=parameters["open_vocab_category_map_file"],
    # )
    semantic_sensor = None

    real_robot = True
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    output_pkl_filename = output_filename + "_" + formatted_datetime + ".pkl"

    robot = HomeRobotZmqClient(
        robot_ip=robot_ip,
        recv_port=recv_port,
        send_port=send_port,
        use_remote_computer=(not local),
        parameters=parameters,
    )
    robot.move_to_nav_posture()
    agent = RobotAgent(
        robot,
        parameters,
        semantic_sensor,
        voxel_map=None,
        enable_realtime_updates=enable_realtime_updates,
    )

    input_path = "kitchen_2024-08-13_17-03-52.pkl"
    # Load map
    input_path = Path(input_path)
    print("Loading:", input_path)

    print("Reading from pkl file of raw observations...")
    agent.get_voxel_map().read_from_pickle(input_path, num_frames=-1)

    prompt = ObjectManipNavPromptBuilder()
    # client = OpenaiClient(prompt)
    client = Gemma2bClient(prompt)

    pos_err_threshold = parameters["trajectory_pos_err_threshold"]
    rot_err_threshold = parameters["trajectory_rot_err_threshold"]

    # Define special skills
    def go_to_landmark(self, object_goal: str):
        landmarks = {
            "cabinet": np.array([0.70500136, 0.34254823, 0.85715184]),
            "bottle": np.array([0.75, -0.18, 1.57324166]),
            "home": np.array([0, 0, 0]),
            "user": np.array([0, 0, 1.57]),
        }
        if object_goal not in landmarks:
            print(f"!!! '{object_goal}' INVALID LANDMARK TO GO TO !!!")
            return False

        current = robot.get_base_pose()
        res = self.planner.plan(current, landmarks[object_goal])
        print("RES: ", res.success)

        if res.success:
            robot.switch_to_navigation_mode()
            print(f"- Going to {object_goal}")
            for i, pt in enumerate(res.trajectory):
                print("-", i, pt.state)

            # Follow the planned trajectory
            robot.execute_trajectory(
                [pt.state for pt in res.trajectory],
                pos_err_threshold=pos_err_threshold,
                rot_err_threshold=rot_err_threshold,
            )
        else:
            print("[ERROR] NO PLAN COULD BE GENERATED")
            return False

        return True

    def pick(self, object_goal: str):
        allowed_objects = {
            "bottle": np.array([0.75, -0.18, 1.57324166]),
        }
        if object_goal not in allowed_objects:
            print(f"!!! '{object_goal}' INVALID OBJECT TO PICK !!!")
            return False

        current = robot.get_base_pose()
        res = self.planner.plan(current, allowed_objects[object_goal])
        print("RES: ", res.success)

        if res.success:
            robot.switch_to_navigation_mode()
            print(f"- Going to {object_goal}")
            for i, pt in enumerate(res.trajectory):
                print("-", i, pt.state)

            # Follow the planned trajectory
            robot.execute_trajectory(
                [pt.state for pt in res.trajectory],
                pos_err_threshold=pos_err_threshold,
                rot_err_threshold=rot_err_threshold,
            )
        else:
            print("[ERROR] NO PLAN COULD BE GENERATED")
            return False

        pickup_policy_path = f"/lerobot/outputs/train/2024-08-07/18-55-17_stretch_real_diffusion_default/checkpoints/100000/pretrained_model"
        pickup_leader = ROS2LfdLeader(
            robot=robot,
            verbose=False,
            data_dir="./data",
            user_name="Jensen",
            task_name="pickup_all_purpose",
            env_name="kitchen",
            save_images=False,
            teleop_mode="base_x",
            record_success=False,
            policy_name="diffusion",
            policy_path=pickup_policy_path,
            device="cuda",
            force_execute=True,
            disable_recording=True,
        )
        try:
            robot.switch_to_manipulation_mode()
            robot.move_to_manip_posture()
            pickup_leader.run()

        except Exception as e:
            print("[ERROR] Bottle pickup failed: ", e)
            return False

        robot.switch_to_navigation_mode()
        robot.move_to_nav_posture()
        return True

    def open_cabinet(self):
        cabinet_policy_path = f"/lerobot/outputs/train/2024-07-28/17-34-36_stretch_real_diffusion_default/checkpoints/100000/pretrained_model"
        cabinet_leader = ROS2LfdLeader(
            robot=robot,
            verbose=False,
            data_dir="./data",
            user_name="Jensen",
            task_name="open_cabinet",
            env_name="kitchen",
            save_images=False,
            teleop_mode="base_x",
            record_success=False,
            policy_name="diffusion",
            policy_path=cabinet_policy_path,
            device="cuda",
            force_execute=True,
            disable_recording=True,
        )
        try:
            robot.switch_to_manipulation_mode()
            robot.move_to_manip_posture()
            # good starting config for cabinet opening
            robot.arm_to(
                [
                    0.0,  # base_x
                    0.8,  # lift
                    0.02,  # arm
                    0.0,  # wrist yaw, pitch, roll
                    -0.8,
                    0.0,
                ],
                gripper=0.6,
                blocking=True,
            )
            cabinet_leader.run()

        except Exception as e:
            print("[ERROR] Opening cabinet failed: ", e)
            return False

        robot.switch_to_navigation_mode()
        robot.move_to_nav_posture()
        return True

    def say(self, msg: str):
        print("[STRETCH SAYS]: ", msg)
        return True

    # Replace with custom methods
    agent.go_to = go_to_landmark.__get__(agent, RobotAgent)
    agent.open_cabinet = open_cabinet.__get__(agent, RobotAgent)
    agent.pick = pick.__get__(agent, RobotAgent)
    agent.say = say.__get__(agent, RobotAgent)

    while True:
        text = input("Enter a long horizon task: ")
        plan = client(text)
        print(f"Generated plan: \n{plan}")
        proceed = input("Proceed with plan? [y/n]: ")

        if plan.startswith("```python"):
            plan = plan.split("\n", 1)[1]

        if plan.endswith("```"):
            plan = plan.rsplit("\n", 1)[0]

        plan += "\nexecute_task(self.go_to, self.pick, self.place, self.say, self.open_cabinet, self.close_cabinet, self.wave, self.get_detections)"

        if proceed != "y":
            print("Exiting...")
            continue

        agent.execute(plan)

    if write_instance_images:
        agent.save_instance_images(".", verbose=True)


if __name__ == "__main__":
    main()
