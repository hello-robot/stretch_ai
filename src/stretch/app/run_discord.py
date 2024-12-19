#!/usr/bin/env python3

# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

from typing import Optional

import click

# import stretch.utils.logger as logger
from stretch.agent.robot_agent import RobotAgent
from stretch.agent.zmq_client import HomeRobotZmqClient
from stretch.core import get_parameters
from stretch.llms import get_llm_choices
from stretch.llms.discord_bot import StretchDiscordBot
from stretch.perception import create_semantic_sensor
from stretch.utils.logger import Logger

logger = Logger(__name__)


@click.command()
@click.option("--robot_ip", default="", help="IP address of the robot")
@click.option("--reset", is_flag=True, help="Reset the robot to origin before starting")
@click.option(
    "--local",
    is_flag=True,
    help="Set if we are executing on the robot and not on a remote computer",
)
@click.option("--parameter_file", default="default_planner.yaml", help="Path to parameter file")
@click.option(
    "--match_method",
    "--match-method",
    type=click.Choice(["class", "feature"]),
    default="feature",
    help="Method to match objects to pick up. Options: class, feature.",
    show_default=True,
)
@click.option(
    "--llm",
    # default="gemma2b",
    # default="qwen25-3B-Instruct",
    default="qwen25",
    help="Client to use for language model. Recommended: gemma2b, openai",
    type=click.Choice(get_llm_choices()),
)
@click.option(
    "--realtime",
    "--real-time",
    "--enable-realtime-updates",
    "--enable_realtime_updates",
    is_flag=True,
    help="Enable real-time updates so that the robot will dynamically update its map",
)
@click.option(
    "--device_id",
    default=0,
    help="ID of the device to use for perception",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Set to print debug information",
)
@click.option(
    "--show_intermediate_maps",
    "--show-intermediate-maps",
    is_flag=True,
    help="Set to visualize intermediate maps",
)
@click.option(
    "--target_object",
    "--target-object",
    default="",
    help="Name of the object to pick up",
)
@click.option(
    "--receptacle",
    default="",
    help="Name of the receptacle to place the object in",
)
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
@click.option(
    "--use_voice",
    "--use-voice",
    is_flag=True,
    help="Set to use voice input",
)
@click.option(
    "--radius",
    default=3.0,
    type=float,
    help="Radius of the circle around initial position where the robot is allowed to go.",
)
@click.option("--open_loop", "--open-loop", is_flag=True, help="Use open loop grasping")
@click.option("--server_ip", "--server-ip", default="127.0.0.1", type=str)
@click.option(
    "--debug_llm",
    "--debug-llm",
    is_flag=True,
    help="Set to print LLM responses to the console, to debug issues when parsing them when trying new LLMs.",
)
@click.option(
    "--task", default="pickup", help="Task to perform", type=click.Choice(["pickup", "dynamem"])
)
@click.option(
    "--token",
    default=None,
    help="The token for the discord bot. Will be read from env if not available.",
)
def main(
    robot_ip: str = "192.168.1.15",
    token: Optional[str] = None,
    local: bool = False,
    parameter_file: str = "config/default_planner.yaml",
    device_id: int = 0,
    verbose: bool = False,
    show_intermediate_maps: bool = False,
    reset: bool = False,
    target_object: str = "",
    receptacle: str = "",
    match_method: str = "feature",
    llm: str = "gemma",
    task: str = "pickup",
    use_voice: bool = False,
    open_loop: bool = False,
    debug_llm: bool = False,
    realtime: bool = False,
    radius: float = 3.0,
    server_ip: str = "127.0.0.1",
    input_path: str = "",
):
    """Set up the robot, create a task plan, and execute it."""
    # Create robot
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

    # Agents wrap the robot high level planning interface for now
    agent = RobotAgent(robot, parameters, semantic_sensor, enable_realtime_updates=realtime)
    print("Starting robot agent: initializing...")
    agent.start(visualize_map_at_start=show_intermediate_maps)
    if reset:
        print("Reset: moving robot to origin")
        agent.move_closed_loop([0, 0, 0], max_time=60.0)

    if radius is not None and radius > 0:
        print("Setting allowed radius to:", radius)
        agent.set_allowed_radius(radius)

    # Load a PKL file from a previous run and process it
    # This will use ICP to match current observations to the previous ones
    # ANd then update the map with the new observations
    if input_path is not None and len(input_path) > 0:
        print("Loading map from:", input_path)
        agent.load_map(input_path)

    # Pass in the information we need to create the task
    bot = StretchDiscordBot(
        agent,
        token,
        llm=llm,
        task=task,
        skip_confirmations=True,
        output_path=".",
        device_id=device_id,
        visual_servo=True,
        kwargs={"match_method": match_method, "mllm_for_visual_grounding": True},
        server_ip=server_ip,
        use_voice=use_voice,
        debug_llm=debug_llm,
    )

    # Start the bot

    @bot.client.command(name="summon", help="Summon the bot to a channel.")
    async def summon(ctx):
        """Summon the bot to a channel."""
        print("Summoning the bot.")
        print(" -> Channel name:", ctx.channel.name)
        print(" -> Channel ID:", ctx.channel.id)
        bot.allowed_channels.visit(ctx.channel)
        await ctx.send("Hello! I am here to help you.")

    obs = robot.get_observation()
    bot.push_task_to_all_channels(content=obs.rgb)
    bot.run()

    # At the end, disable everything
    robot.stop()


if __name__ == "__main__":
    main()
