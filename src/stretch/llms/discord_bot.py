# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import datetime
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import discord
from termcolor import colored

# import stretch.utils.logger as logger
from stretch.agent.robot_agent import RobotAgent
from stretch.agent.task.dynamem import DynamemTaskExecutor
from stretch.agent.task.pickup import PickupExecutor
from stretch.llms import PickupPromptBuilder, get_llm_client
from stretch.utils.discord_bot import DiscordBot, Task
from stretch.utils.logger import Logger

logger = Logger(__name__)


class StretchDiscordBot(DiscordBot):
    """Simple stretch discord bot. Connects to Discord via the API."""

    def __init__(
        self,
        agent: RobotAgent,
        token: Optional[str] = None,
        llm: str = "qwen25",
        task: str = "pickup",
        skip_confirmations: bool = False,
        output_path: str = ".",
        device_id: int = 0,
        visual_servo: bool = True,
        server_ip: str = "127.0.0.1",
        use_voice: bool = False,
        debug_llm: bool = False,
        manipulation_only: bool = False,
        kwargs: Dict[str, Any] = None,
        home_channel: str = "talk-to-stretch",
    ) -> None:
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
        self.robot = self.agent.robot
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
        # TODO: we should have an Executor abstract class here!
        if self.task == "pickup":
            self.executor = PickupExecutor(
                robot,
                agent,
                available_actions=prompt.get_available_actions(),
                dry_run=False,
                discord_bot=self,
            )  # type: ignore
        elif self.task == "dynamem":
            self.executor = DynamemTaskExecutor(
                robot,
                agent.parameters,
                visual_servo=visual_servo,
                match_method=kwargs["match_method"],
                device_id=device_id,
                output_path=output_path,
                server_ip=server_ip,
                skip_confirmations=skip_confirmations,
                mllm=kwargs["mllm_for_visual_grounding"],
                manipulation_only=manipulation_only,
                discord_bot=self,
            )  # type: ignore
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

        # Plans list
        self.next_plan = None
        self._plan_lock = threading.Lock()
        self._plan_thread = None

        print(self.allowed_channels)

        # PRINTS HOW MANY GUILDS / SERVERS THE BOT IS IN.
        print("This bot is in " + str(guild_count) + " guild(s).")

        print("Starting the message processing queue.")
        self.process_queue.start()

        # Start the plan thread
        self.start_plan_thread()

    def push_task_to_all_channels(
        self, message: Optional[str] = None, content: Optional[str] = None
    ):
        """Push a task to all channels. Message will be "as-is" with no processing.

        Args:
            message: The message to send to the channel.
            content: The content (image string) to send to the channel.
        """

        for channel in self.allowed_channels:
            self.push_task(channel, message=message, content=content, explicit=True)

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
        # TODO: Do we ever want to add the channel name? If so we can revert this change
        # text = f"{sender_name} on #{channel_name}: " + message.content
        text = f"{sender_name}: " + message.content
        self.push_task(channel=message.channel, message=text)

        print("Current task queue: ", self.task_queue.qsize())
        # print(" -> Response:", response)
        return None

    async def handle_task(self, task: Task):
        """Handle a task by sending the message to the channel. This will make the necessary calls in its thread to the different child functions that send messages, for example."""
        print()
        print("-" * 40)
        print("Handling task from channel:", task.channel.name)
        print("Handling task: message = \n", task.message)

        text = task.message
        try:
            if task.explicit:
                print("This task was explicitly triggered.")
                await task.channel.send(task.message)
                if task.content is not None:
                    # Filename is computed from date and time
                    filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".png"
                    await task.channel.send(file=discord.File(task.content, filename=filename))
                return
        except Exception as e:
            print(colored("Error in handling task: " + str(e), "red"))

        with self._llm_lock:
            response = self.llm_client(text, verbose=True)
            print("Response:", response)
            parsed_response = self.prompt.parse_response(response)
            print("Parsed response:", parsed_response)
            self.add_robot_plan(parsed_response, channel=task.channel)

    def add_robot_plan(self, response: List[Tuple[str, str]], channel: discord.TextChannel):
        """Add a task to the task queue."""
        with self._plan_lock:
            self.next_plan = response, channel

    def plan_thread(self):
        """Loop. Check to see if next plan received. If so, execute it. Else, sleep. After execution, set next_plan to None."""

        while self.robot.is_running:
            if self.next_plan is not None:
                with self._plan_lock:
                    response, channel = self.next_plan
                    self.next_plan = None
                self.executor(response, channel=channel)
            else:
                time.sleep(0.01)

    def start_plan_thread(self):
        """Start the plan thread."""
        self._plan_thread = threading.Thread(target=self.plan_thread)
        self._plan_thread.start()

    def __del__(self):
        """Destructor. Stop the plan thread."""
        if self._plan_thread is not None:
            self._plan_thread.join()
