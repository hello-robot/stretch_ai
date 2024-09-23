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
import pprint

import click

# Mapping and perception
from stretch.agent.robot_agent import RobotAgent
from stretch.agent.task.llm_plan import LLMPlanTask
from stretch.agent.zmq_client import HomeRobotZmqClient
from stretch.core import get_parameters
from stretch.llms.openai_client import OpenaiClient
from stretch.llms.prompts import ObjectManipNavPromptBuilder
from stretch.perception import create_semantic_sensor


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
@click.option("--target_object", type=str, default="toy", help="Type of object to pick up and move")
def main(
    device_id: int = 0,
    verbose: bool = True,
    parameter_file: str = "config/default_planner.yaml",
    local: bool = True,
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
    target_object: str = "toy",
):

    print("- Load parameters")
    parameters = get_parameters(parameter_file)
    semantic_sensor = create_semantic_sensor(
        device_id=device_id,
        verbose=verbose,
        category_map_file=parameters["open_vocab_category_map_file"],
    )

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
    agent = RobotAgent(robot, parameters, semantic_sensor)
    agent.update()
    # agent.voxel_map.read_from_pickle(input_file)

    prompt = ObjectManipNavPromptBuilder()
    client = OpenaiClient(prompt)

    print("Starting robot exploration...")

    agent.run_exploration(
        rate=10,
        manual_wait=False,
        explore_iter=20,
        task_goal=target_object,  # arbitrary object to collect
        # as many instances as possible
        go_home_at_end=True,
        visualize=False,
    )

    while True:
        text = input("Enter a long horizon task: ")
        plan = client(text)
        print(f"Generated plan: \n{plan}")
        proceed = input("Proceed with plan? [y/n]: ")

        if plan.startswith("```python"):
            plan = plan.split("\n", 1)[1]

        if plan.endswith("```"):
            plan = plan.rsplit("\n", 1)[0]

        llm_plan_task = LLMPlanTask(agent, plan)
        plan = llm_plan_task.get_task()
        pprint.pprint(plan)

        if proceed != "y":
            print("Exiting...")
            continue

        plan.run()
        break


if __name__ == "__main__":
    main()
