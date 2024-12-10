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
import click

# Mapping and perception
from stretch.agent.robot_agent import RobotAgent
from stretch.agent.task.llm_plan import LLMPlanExecutor
from stretch.agent.zmq_client import HomeRobotZmqClient
from stretch.core import get_parameters
from stretch.llms import get_llm_client
from stretch.llms.prompts import ObjectManipNavPromptBuilder
from stretch.perception import create_semantic_sensor
from stretch.utils import logger

logger = logger.Logger(__name__)


@click.command()
@click.option("--local", is_flag=True, help="Run code locally on the robot.")
@click.option("--robot_ip", default="")
@click.option("--device-id", default=0, help="Device ID for the semantic sensor")
@click.option("--llm", default="qwen25-3B-Instruct", help="Language model to use")
@click.option(
    "--input-path",
    "-i",
    "--input_file",
    "--input-file",
    "--input",
    "--input_path",
    type=click.Path(),
    default="",
    help="Path to a saved datafile from a previous exploration of the world.",
)
@click.option("--verbose", default=True)
@click.option("--parameter-file", default="default_planner.yaml")
@click.option("--task", default="", help="Default task to perform")
@click.option("-y", "--yes", is_flag=True, help="Skip confirmation")
@click.option(
    "--enable_realtime_updates",
    "--enable-realtime-updates",
    is_flag=True,
    help="Enable real-time updates so that the robot will dynamically update the map as it moves",
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
    enable_realtime_updates: bool = False,
    input_path: str = "",
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

    if not parameters.get("agent/use_realtime_updates") or enable_realtime_updates:
        logger.error("Real-time updates are required for this demo. Enabling them.")

    agent = RobotAgent(robot, parameters, semantic_sensor, enable_realtime_updates=True)
    agent.start()

    if input_path is not None and len(input_path) > 0:
        print("Loading map from:", input_path)
        agent.load_map(input_path)

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
            proceed = "y"
        else:
            proceed = input("Proceed with plan? [y/n]: ")

        if plan.startswith("```python"):
            plan = plan.split("\n", 1)[1]

        if plan.endswith("```"):
            plan = plan.rsplit("\n", 1)[0]

        llm_plan_executor = LLMPlanExecutor(agent, plan)

        if proceed == "y":
            try:
                llm_plan_executor.run()
            except Exception as e:
                print(f"Error executing plan: {e}")

        # Reset variables
        task = ""
        plan = ""
        proceed = ""
        llm_plan_executor = None


if __name__ == "__main__":
    main()
