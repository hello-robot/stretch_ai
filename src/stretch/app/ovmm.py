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
import pprint

import click

# Mapping and perception
from stretch.agent.robot_agent import RobotAgent
from stretch.agent.task.llm_plan import LLMPlanTask
from stretch.agent.zmq_client import HomeRobotZmqClient
from stretch.core import get_parameters
from stretch.llms import get_llm_client
from stretch.llms.prompts import ObjectManipNavPromptBuilder
from stretch.perception import create_semantic_sensor


@click.command()
@click.option("--local", is_flag=True, help="Run code locally on the robot.")
@click.option("--robot_ip", default="")
@click.option("--device-id", default=0, help="Device ID for the semantic sensor")
@click.option("--llm", default="qwen25-3B-Instruct", help="Language model to use")
@click.option("--verbose", default=True)
@click.option("--parameter-file", default="default_planner.yaml")
@click.option("--task", default="", help="Default task to perform")
@click.option("-y", "--yes", is_flag=True, help="Skip confirmation")
@click.option(
    "--disable_realtime_updates",
    "--disable-realtime-updates",
    is_flag=True,
    help="Disable real-time updates",
)
def main(
    local: bool = True,
    robot_ip: str = "",
    device_id: int = 0,
    llm: str = "qwen25-3B-Instruct",
    verbose: bool = True,
    parameter_file: str = "config/default_planner.yaml",
    task: str = "",
    yes: bool = False,
    disable_realtime_updates: bool = False,
):
    parameters = get_parameters(parameter_file)
    robot = HomeRobotZmqClient(
        robot_ip=robot_ip,
        use_remote_computer=(not local),
        parameters=parameters,
    )
    semantic_sensor = create_semantic_sensor(
        parameters=parameters,
        device_id=device_id,
        verbose=verbose,
    )
    agent = RobotAgent(
        robot, parameters, semantic_sensor, enable_realtime_updates=not disable_realtime_updates
    )
    agent.start()

    prompt = ObjectManipNavPromptBuilder()
    client = get_llm_client(llm, prompt=prompt)

    while robot.running:

        # Get a plan from the language model
        if task:
            text = task
        else:
            text = input("Enter a long horizon task: ")
        plan = client(text)
        print(f"Generated plan: \n{plan}")

        if yes:
            proceed = True
        else:
            proceed = input("Proceed with plan? [y/n]: ")

        if plan.startswith("```python"):
            plan = plan.split("\n", 1)[1]

        if plan.endswith("```"):
            plan = plan.rsplit("\n", 1)[0]

        llm_plan_task = LLMPlanTask(agent, plan)
        plan = llm_plan_task.get_task()
        pprint.pprint(plan)

        if yes:
            proceed = "y"

        if proceed != "y":
            print("Exiting...")
            continue

        try:
            plan.run()
        except Exception as e:
            print(f"Error executing plan: {e}")

        if task:
            robot.stop()
            break


if __name__ == "__main__":
    main()
