#!/usr/bin/env python3

# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import threading
from typing import Any, Dict, Optional

import click

# import stretch.utils.logger as logger
from stretch.agent.robot_agent import RobotAgent
from stretch.agent.task.pickup import PickupExecutor
from stretch.agent.zmq_client import HomeRobotZmqClient
from stretch.core import get_parameters
from stretch.llms import LLMChatWrapper, PickupPromptBuilder, get_llm_choices, get_llm_client
from stretch.perception import create_semantic_sensor
from stretch.utils.discord_bot import DiscordBot
from stretch.utils.logger import Logger
from stretch.utils.image import numpy_image_to_bytes

import discord

logger = Logger(__name__)

class StretchDiscordBot(DiscordBot):
    """Simple stretch discord bot. Connects to Discord via the API."""
    
    def __init__(self, agent: RobotAgent, token: Optional[str] = None, llm: str = "qwen25", task: str = "pickup", skip_confirmations: bool = False, output_path: str = ".", device_id: int = 0, visual_servo: bool = True, server_ip: str = "127.0.0.1", use_voice: bool = False, debug_llm: bool = False, manipulation_only: bool = False, kwargs: Dict[str, Any] = None, home_channel: str = "talk-to-stretch") -> None:
        """
        Create a new Discord bot that can interact with the robot.

        Args:
            agent: The robot agent that will be used to control the robot.
            token: The token for the discord bot. Will be read from env if not available.
            llm: The language model to use.
            task: The task to perform. Currently only "pickup" is supported.
            skip_confirmations: Set to skip confirmations from the user.
            output_path: The path to save output files.
            device_id: The ID of the device to use for perception.
            visual_servo: Set to use visual servoing.
            kwargs: Additional parameters.

        Returns:
            None
        """
        super().__init__(token)
        robot = agent.robot

        # Create the prompt we will use to control the robot
        prompt = PickupPromptBuilder()

        # Save the parameters
        self.task = task
        self.agent = agent
        self.parameters = agent.parameters
        self.visual_servo = visual_servo
        self.device_id = device_id
        self.output_path = output_path
        self.server_ip = server_ip
        self.skip_confirmations = skip_confirmations
        self.kwargs = kwargs
        self.prompt = prompt

        # LLM info
        self.home_channel = home_channel
        self.sent_prompt = False

        if kwargs is None:
            # Default parameters
            kwargs = {
                    "match_method": "feature",
                    "mllm_for_visual_grounding": False,
                    }

        # Executor handles outputs from the LLM client and converts them into executable actions
        if self.task == "pickup":
            self.executor = PickupExecutor(
                robot, agent, available_actions=prompt.get_available_actions(), dry_run=False
            )
        elif self.task == "dynamem":
            executor = DynamemTaskExecutor(
                robot,
                parameters,
                visual_servo=visual_servo,
                match_method=kwargs["match_method"],
                device_id=device_id,
                output_path=output_path,
                server_ip=server_ip,
                skip_confirmations=skip_confirmations,
                mllm=kwargs["mllm_for_visual_grounding"],
                manipulation_only=manipulation_only,
            )
        else:
            raise NotImplementedError(f"Task {task} is not implemented.")

        # Get the LLM client
        self.llm_client = get_llm_client(llm, prompt=prompt)

        self._llm_lock = threading.Lock()

    def on_ready(self):
        """Event listener called when the bot has switched from offline to online."""
        print(f"{self.client.user} has connected to Discord!")
        guild_count = 0

        print("Bot User name:", self.client.user.name)
        print("Bot Global name:", self.client.user.global_name)
        print("Bot User IDL", self.client.user.id)
        self._user_name = self.client.user.name
        self._user_id = self.client.user.id

        # This is from https://builtin.com/software-engineering-perspectives/discord-bot-python
        # LOOPS THROUGH ALL THE GUILD / SERVERS THAT THE BOT IS ASSOCIATED WITH.
        for guild in self.client.guilds:
            # PRINT THE SERVER'S ID AND NAME.
            print(f"Joining Server {guild.id} (name: {guild.name})")

            # INCREMENTS THE GUILD COUNTER.
            guild_count = guild_count + 1

            for channel in guild.text_channels:
                if channel.name == self.home_channel:
                    print(f"Adding home channel {channel} to the allowed channels.")
                    self.allowed_channels.add_home(channel)
                    break

        print(self.allowed_channels)

        # PRINTS HOW MANY GUILDS / SERVERS THE BOT IS IN.
        print("This bot is in " + str(guild_count) + " guild(s).")

        print("Starting the message processing queue.")
        self.process_queue.start()

    def process(self):
        with self._llm_lock:
            llm_response = self.chat_wrapper.query(verbose=debug_llm)
            if debug_llm:
                print("Parsed LLM Response:", llm_response)

            ok = executor(llm_response)

    def push_task_to_all_channels(self, message: Optional[str], content: Optional[str] = None):
        """Push a task to all channels."""
        for channel in self.allowed_channels:
            self.push_task(channel, message=message, content=content)

    def on_message(self, message: discord.Message, verbose: bool = False):
        """Event listener for whenever a new message is sent to a channel that this bot is in."""
        if verbose:
            # Printing some information to learn about what this actually does
            print(message)
            print("Content =", message.content)
            print("Content type =", type(message.content))
            print("Author name:", message.author.name)
            print("Author global name:", message.author.global_name)

        # This is your actual username
        # sender_name = message.author.name
        sender_name = message.author.display_name
        # Only necessary once we want multi-server Friends
        # global_name = message.author.global_name

        # Skip anything that's from this bot
        if message.author.id == self._user_id:
            return None

        # TODO: make this a command line parameter for which channel(s) he should be in
        channel_name = message.channel.name
        print("Channel name:", channel_name)
        channel_id = message.channel.id
        print("Channel ID:", channel_id)
        # datetime = message.created_at

        timestamp = message.created_at.timestamp()
        print("Timestamp:", timestamp)

        print(self.allowed_channels)
        if not message.channel in self.allowed_channels:
            print(" -> Not in allowed channels. Skipping.")
            return None

        # Construct the text to prompt the AI
        text = f"{sender_name} on #{channel_name}: " + message.content
        self.push_task(channel=message.channel, message=text)

        print("Current task queue: ", self.task_queue.qsize())
        # print(" -> Response:", response)
        return None


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
    default="qwen25-3B-Instruct",
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
@click.option("--token", default=None, help="The token for the discord bot. Will be read from env if not available.")
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
    bot = StretchDiscordBot(agent, token, llm=llm, task="pickup", skip_confirmations=True, output_path=".", device_id=device_id, visual_servo=True, kwargs={"match_method": match_method}, server_ip=server_ip, use_voice=use_voice, debug_llm=debug_llm)

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
