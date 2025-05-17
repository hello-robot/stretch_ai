# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import click

from stretch.demo.robot_agent_sender import RobotAgent
from stretch.agent.task.dynamem import EQAExecuter
from stretch.agent.zmq_client import HomeRobotZmqClient

# Mapping and perception
from stretch.core.parameters import get_parameters


@click.command()
# by default you are running these codes on your workstation, not on your robot.
@click.option(
    "--robot_ip",
    "--robot-ip",
    default="127.0.0.1",
    type=str,
    help="Robot IP address (leave empty for saved default)",
)
@click.option(
    "--not_rotate_in_place",
    "-N",
    is_flag=True,
    help="Whether the robot rotates in place at the beginning",
)
@click.option(
    "--discord",
    "-D",
    is_flag=True,
    help="Whether we would launch discord bot",
)
@click.option(
    "--save_rerun",
    "--SR",
    is_flag=True,
    help="Whether we should save rerun rrd",
)
def main(
    robot_ip,
    discord: bool = False,
    not_rotate_in_place: bool = False,
    save_rerun: bool = False,
    **kwargs,
):
    """
    Including only some selected arguments here.
    """
    click.echo("Will connect to a Stretch robot and collect a short trajectory.")
    robot = HomeRobotZmqClient(robot_ip=robot_ip)

    print("- Load parameters")
    parameters = get_parameters("dynav_config.yaml")
    robot.move_to_nav_posture()
    robot.set_velocity(v=30.0, w=15.0)

    parameters["encoder"] = None

    print("- Start robot agent with data collection")
    agent = RobotAgent(robot, parameters, save_rerun=save_rerun)
    agent.start()

    if discord:
        # Discord is not installed by default, import only needed
        from stretch.llms.discord_bot import StretchDiscordBot

        bot = StretchDiscordBot(agent, task="eqa")
        if not not_rotate_in_place:
            bot.executor.rotate_in_place()

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
    else:

        executor = EQAExecuter(agent)

        if not not_rotate_in_place:
            executor.rotate_in_place()

        while True:

            # If target object and receptacle are provided, set mode to manipulation
            question = input("Question (Pess enter to quit):").lower()
            if question.replace(" ", "") == "":
                break

            robot.move_to_nav_posture()
            robot.switch_to_navigation_mode()
            robot.say("Answering the question " + question)
            executor(question)


if __name__ == "__main__":
    main()
