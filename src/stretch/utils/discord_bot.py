# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright 2024 Chris Paxton
# From https://github.com/cpaxton/virgil/blob/main/virgil/io/discord_bot.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# (c) 2024 by Chris Paxton

import asyncio
import io
import os
import queue
import threading
import timeit
from dataclasses import dataclass
from itertools import chain
from typing import Dict, List, Optional

import discord
from discord.ext import commands, tasks
from termcolor import colored


def read_discord_token_from_env():
    """Helpful tool to get a discord token from the command line, e.g. for a bot."""
    from dotenv import load_dotenv

    # Load environment variables from .env file
    load_dotenv()
    TOKEN = os.getenv("DISCORD_TOKEN")
    if not TOKEN:
        raise ValueError("DISCORD_TOKEN environment variable not set.")
    return TOKEN


@dataclass
class Task:
    message: discord.Message
    channel: discord.TextChannel
    content: str
    explicit: bool = False
    t: float = None


class ChannelList:
    """Tracks the channels that the bot is allowed to post in."""

    def __init__(self):
        self.home_channels: List[discord.TextChannel] = []
        self.visiting_channels: Dict[
            discord.TextChannel, float
        ] = {}  # Maps channel to expiration time

    def add_home(self, channel: discord.TextChannel):
        """Add a channel to the home list."""
        self.home_channels.append(channel)

    def remove_home(self, channel: discord.TextChannel):
        """Remove a channel from the home list."""
        self.home_channels = [c for c in self.home_channels if c != channel]

    def visit(self, channel: discord.TextChannel, timeout_s: float = 60):
        """Add a channel to the visiting list."""
        self.visiting_channels[channel] = timeit.default_timer() + timeout_s

    def is_valid(self, channel: discord.TextChannel):
        """Check if a channel is valid to post in."""
        if channel in self.home_channels:
            return True
        elif expiration := self.visiting_channels.get(channel):
            if expiration > timeit.default_timer():
                return True
            print(f"Visit to channel has expired. {channel}")
            del self.visiting_channels[channel]
        return False

    def __contains__(self, channel: discord.TextChannel):
        return self.is_valid(channel)

    def __iter__(self):
        return chain(self.home_channels, (vc.channel for vc in self.visiting_channels.values()))

    def __len__(self):
        return len(self.home_channels) + len(self.visiting_channels)

    def __str__(self) -> str:
        return f"Home: {self.home_channels}\nVisiting: {self.visiting_channels.keys()}"


class DiscordBot:
    def __init__(self, token: Optional[str] = None, timeout: float = 180):
        """Create the bot, using an authorization token from Discord."""
        # Create intents
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True

        # This is how long until we drop tasks
        self.timeout = timeout

        # Track if we have sent our startup message
        self._started = False

        # Store the user ID and name for the bot
        self._user_id = None
        self._user_name = None

        # Create an instance of a Client
        # self.client = discord.Client(intents=intents)
        self.client = commands.Bot(command_prefix="!", intents=intents)
        self._setup_hooks(self.client)

        # Save the token
        if token is None:
            token = read_discord_token_from_env()
        self.token = token

        self.running = True
        self.task_queue: queue.Queue = queue.Queue()
        self.queue_lock = threading.Lock()
        self.allowed_channels = ChannelList()

    def push_task(
        self,
        channel,
        message: Optional[str] = None,
        content: Optional[str] = None,
        explicit: bool = False,
    ):
        """Add a message to the queue to send."""
        # print("Adding task to queue:", message, channel.name, content)
        self.task_queue.put(
            Task(message, channel, content, explicit=explicit, t=timeit.default_timer())
        )
        # print( "Queue length after push:", self.task_queue.qsize())

    def send_message(self, channel, message: Optional[str] = None, content: Optional[str] = None):
        """Send a message to a channel.

        Args:
            channel: The channel to send the message to.
            message: The message to send (text), if any.
            content: The image or file to send as a BytesIO object.
        """
        self.push_task(channel, message=message, content=content, explicit=True)

    async def handle_task(self, task: Task):
        """Handle a task by sending the message to the channel. This will make the necessary calls in its thread to the different child functions that send messages, for example."""
        print()
        print(
            "Handling task: message = ",
            task.message,
            " channel = ",
            task.channel.name,
            " content = ",
            task.content,
        )
        if task.message is not None:
            print(" - Sending message:", task.message)
            await task.channel.send(task.message)
        if task.content is not None:
            # Send an image
            print(" - Sending content:", task.content)
            # This should be a Discord file
            # Create a BytesIO object
            byte_arr = io.BytesIO()

            image = task.content
            # Save the image to the BytesIO object
            image.save(byte_arr, format="PNG")  # Save as PNG
            print(" - Image saved to byte array, format: ", image.format)

            # Move the cursor to the beginning of the BytesIO object
            byte_arr.seek(0)

            file = discord.File(byte_arr, filename="image.png")
            await task.channel.send(file=file)

    @tasks.loop(seconds=0.1)
    async def process_queue(self):
        """Process the queue of messages to send."""

        if not self._started:
            # Loop over all channels we have not yet started
            # Add a message for each one
            for channel in self.client.get_all_channels():
                if channel.type == discord.ChannelType.text:
                    if channel in self.allowed_channels:
                        print(f"Introducing myself to channel {channel.name}")
                        try:
                            self.push_task(
                                channel, message=self.greeting(), content=None, explicit=True
                            )
                        except Exception as e:
                            print(colored("Error in introducing myself: " + str(e), "red"))
            self._started = True

        # Print queue length
        # print("Queue length:", self.task_queue.qsize())
        try:
            task = self.task_queue.get_nowait()

            # Peak at the next task
            if self.task_queue.qsize() > 0:
                print(
                    "Next task:",
                    self.task_queue.queue[0].message,
                    self.task_queue.queue[0].channel.name,
                )

            # While the channel is the same and content is None...
            while (
                self.task_queue.qsize() > 0
                and self.task_queue.queue[0].channel == task.channel
                and self.task_queue.queue[0].content is None
            ):
                # Pop the next task
                extra_task = self.task_queue.get_nowait()
                print("Popped task:", extra_task.message, extra_task.channel.name)

                # Add this message to the current task message
                task.message += "\n" + extra_task.message

            if task.t + self.timeout < timeit.default_timer():
                print("Dropping task due to timeout: ", task.message, task.channel.name)
                return

            print("Handling task from queue:", task)
            await self.handle_task(task)
        except queue.Empty:
            await asyncio.sleep(0.1)  # Wait a bit before checking again
        except Exception as e:
            print(colored("Error in processing queue: " + str(e), "red"))
            raise e

    def greeting(self) -> str:
        """Return a greeting message."""
        return "Hello everyone!"

    def _setup_hooks(self, client: discord.Client):
        """Prepare the various hooks to use this Bot object's methods."""

        @client.event
        async def on_ready():
            self._started = False
            self._user_name = self.client.user.name
            self._user_id = self.client.user.id
            return self.on_ready()

        @client.event
        async def on_message(message: discord.Message):
            # This line is important to allow commands to work
            # await bot.process_commands(message)

            # Check if the bot was mentioned
            # print()
            # print("Mentions:", message.mentions)
            idx1 = message.content.find("@" + self._user_name)
            idx2 = message.content.find("<@" + str(self._user_id) + ">")
            if idx1 >= 0 or idx2 >= 0:
                print(
                    " ->",
                    self._user_name,
                    " was mentioned in channel",
                    message.channel.name,
                    "with content:",
                    message.content,
                )
                if message.channel not in self.allowed_channels:
                    self.allowed_channels.visit(message.channel)

            print("Message content:", message.content)
            response = self.on_message(message)

            if response is not None:
                print("Sending response:", response)
                await message.channel.send(response)
                print("Done")

            print("-------------")

    def on_ready(self):
        """Event listener called when the bot has switched from offline to online."""
        print(f"{self.client.user} has connected to Discord!")
        guild_count = 0

        print("Bot User name:", self.client.user.name)
        print("Bot Global name:", self.client.user.global_name)

        # This is from https://builtin.com/software-engineering-perspectives/discord-bot-python
        # LOOPS THROUGH ALL THE GUILD / SERVERS THAT THE BOT IS ASSOCIATED WITH.
        for guild in self.client.guilds:
            # PRINT THE SERVER'S ID AND NAME.
            print(f"- {guild.id} (name: {guild.name})")

            # INCREMENTS THE GUILD COUNTER.
            guild_count = guild_count + 1

        # PRINTS HOW MANY GUILDS / SERVERS THE BOT IS IN.
        print("This bot is in " + str(guild_count) + " guilds.")

        print("Starting the message processing queue.")
        self.process_queue.start()

    def on_message(self, message, verbose: bool = False):
        """Event listener for whenever a new message is sent to a channel that this bot is in."""
        if verbose:
            # Printing some information to learn about what this actually does
            print(message)
            print("Content =", message.content)
            print("Content type =", type(message.content))

        text = message.content
        if text.startswith("hello"):
            return "Hello!"

    def run(self):
        # Start the message thread to process the queue
        self.running = True

        async def _main():
            async with self.client as bot:
                await bot.start(self.token)

        # self.client.start(self.token)
        asyncio.run(_main())

    def __del__(self):
        self.running = False


if __name__ == "__main__":
    bot = DiscordBot()
    bot.run()
