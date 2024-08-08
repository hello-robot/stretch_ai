#!/usr/bin/env python3

import stretch.utils.logger as logger
from stretch.agent.operations import WaveOperation
from stretch.agent.robot_agent import RobotAgent
from stretch.agent.task.emote import EmoteManager
from stretch.agent.zmq_client import HomeRobotZmqClient
from stretch.core import get_parameters


def main(
    robot_ip: str = "",
    local: bool = False,
    parameter_file: str = "default_planner.yaml",
):
    # Create robot client
    parameters = get_parameters(parameter_file)
    robot = HomeRobotZmqClient(
        robot_ip=robot_ip,
        use_remote_computer=(not local),
        parameters=parameters,
    )

    robot.move_to_nav_posture()

    # create robot agent
    demo = RobotAgent(robot, parameters=parameters)

    wave = WaveOperation("emote", robot=robot)
    res = wave()
    if not res:
        logger.error("Wave operation failed")

    # Turn off the robot at the end
    robot.stop()


if __name__ == "__main__":
    main()
