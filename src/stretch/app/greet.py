#!/usr/bin/env python3

from stretch.agent.operations import TestOperation
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

    # create robot agent
    demo = RobotAgent(robot, parameters=parameters)

    # create task manager
    manager = EmoteManager(demo)
    task = manager.get_task(TestOperation("emote", manager))

    # run task
    task.run()


if __name__ == "__main__":
    main()
