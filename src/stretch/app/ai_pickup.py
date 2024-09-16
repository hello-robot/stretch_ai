#!/usr/bin/env python3

# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import click

# import stretch.utils.logger as logger
from stretch.agent.robot_agent import RobotAgent
from stretch.agent.task.pickup import PickupTask
from stretch.agent.zmq_client import HomeRobotZmqClient
from stretch.core import get_parameters
from stretch.llms import LLMChatWrapper, PickupPromptBuilder, get_llm_choices, get_llm_client
from stretch.perception import create_semantic_sensor


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
    "--mode",
    default="one_shot",
    help="Mode of operation for the robot.",
    type=click.Choice(["one_shot", "all"]),
)
@click.option(
    "--llm",
    default="gemma2b",
    help="Client to use for language model.",
    type=click.Choice(get_llm_choices()),
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
    "--use_llm",
    "--use-llm",
    is_flag=True,
    help="Set to use the language model",
)
@click.option(
    "--use_voice",
    "--use-voice",
    is_flag=True,
    help="Set to use voice input",
)
@click.option("--open_loop", "--open-loop", is_flag=True, help="Use open loop grasping")
def main(
    robot_ip: str = "192.168.1.15",
    local: bool = False,
    parameter_file: str = "config/default_planner.yaml",
    device_id: int = 0,
    verbose: bool = False,
    show_intermediate_maps: bool = False,
    reset: bool = False,
    target_object: str = "",
    receptacle: str = "",
    mode: str = "one_shot",
    match_method: str = "feature",
    llm: str = "gemma",
    use_llm: bool = False,
    use_voice: bool = False,
    open_loop: bool = False,
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

    # Start moving the robot around
    grasp_client = None

    # Agents wrap the robot high level planning interface for now
    agent = RobotAgent(robot, parameters, semantic_sensor, grasp_client=grasp_client)
    agent.start(visualize_map_at_start=show_intermediate_maps)
    if reset:
        agent.move_closed_loop([0, 0, 0], max_time=60.0)

    prompt = PickupPromptBuilder()

    # Get the LLM client
    llm_client = None
    if use_llm:
        llm_client = get_llm_client(llm, prompt=prompt)
        chat_wrapper = LLMChatWrapper(llm_client, prompt=prompt, voice=use_voice)

    # Parse things and listen to the user
    while robot.running:
        agent.reset()

        say_this = None
        if llm_client is None:
            # Call the LLM client and parse
            if len(target_object) == 0:
                target_object = input("Enter the target object: ")
            if len(receptacle) == 0:
                receptacle = input("Enter the target receptacle: ")
        else:
            # Call the LLM client and parse
            llm_response = chat_wrapper.query()
            target_object = prompt.get_object(llm_response)
            receptacle = prompt.get_receptacle(llm_response)
            say_this = prompt.get_say_this(llm_response)
            # print("LLM response:", llm_response)
            # print("Target object:", target_object)
            # print("Receptacle:", receptacle)
            # print("Say this:", say_this)

        if say_this is not None:
            chat_wrapper.say(say_this)
            # agent.say(say_this)

        if len(target_object) == 0 or len(receptacle) == 0:
            # logger.error("You need to enter a target object and receptacle")
            continue

        # After the robot has started...
        try:
            pickup_task = PickupTask(
                agent,
                target_object=target_object,
                target_receptacle=receptacle,
                matching=match_method,
                use_visual_servoing_for_grasp=not open_loop,
            )
            task = pickup_task.get_task(add_rotate=True, mode=mode)
        except Exception as e:
            print(f"Error creating task: {e}")
            robot.stop()
            raise e

        # Execute the task
        task.run()

        if reset:
            # Send the robot home at the end!
            agent.go_home()

        break

    # At the end, disable everything
    robot.stop()


if __name__ == "__main__":
    main()
