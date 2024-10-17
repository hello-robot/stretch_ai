# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# (c) 2024 Hello Robot by Chris Paxton
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import datetime
from typing import Optional

import click

# Mapping and perception
from stretch.agent.robot_agent import RobotAgent
from stretch.agent.zmq_client import HomeRobotZmqClient
from stretch.core import AbstractRobotClient, Parameters, get_parameters
from stretch.perception import create_semantic_sensor


@click.command()
@click.option("--local", is_flag=True, help="Run code locally on the robot.")
@click.option("--robot_ip", default="")
@click.option("--visualize", default=False, is_flag=True)
@click.option("--manual-wait", default=False, is_flag=True)
@click.option("--output-filename", default="stretch_output", type=str)
@click.option("--show-intermediate-maps", default=False, is_flag=True)
@click.option("--show-final-map", default=False, is_flag=True)
@click.option("--show-paths", default=False, is_flag=True)
@click.option("--random-goals", default=False, is_flag=True)
@click.option(
    "--explore-iter",
    "--explore_iter",
    default=-1,
    help="Number of exploration steps, i.e. times the robot will try to move to an unexplored frontier",
)
@click.option("--navigate-home", default=False, is_flag=True)
@click.option("--force-explore", default=False, is_flag=True)
@click.option("--no-manip", default=False, is_flag=True)
@click.option("--device-id", default=0, help="Device ID for the semantic sensor")
@click.option(
    "--write-instance-images",
    default=False,
    is_flag=True,
    help="write out images of every object we found",
)
@click.option("--parameter-file", default="default_planner.yaml")
@click.option("--reset", is_flag=True, help="Reset the robot to origin before starting")
@click.option(
    "--enable-realtime-updates",
    "--enable_realtime-updates",
    is_flag=True,
    help="Enable real-time updates so the robot will scan its environment and update the map as it moves around",
)
@click.option("--save", is_flag=True, help="Save the map to memory")
def main(
    visualize,
    manual_wait,
    output_filename,
    navigate_home: bool = False,
    device_id: int = 0,
    verbose: bool = True,
    show_intermediate_maps: bool = False,
    show_final_map: bool = False,
    show_paths: bool = False,
    random_goals: bool = True,
    force_explore: bool = False,
    no_manip: bool = False,
    explore_iter: int = 10,
    write_instance_images: bool = False,
    parameter_file: str = "config/default_planner.yaml",
    local: bool = True,
    robot_ip: str = "192.168.1.15",
    reset: bool = False,
    enable_realtime_updates: bool = False,
    save: bool = True,
    **kwargs,
):

    print("- Load parameters")
    parameters = get_parameters(parameter_file)

    robot = HomeRobotZmqClient(
        robot_ip=robot_ip,
        use_remote_computer=(not local),
        parameters=parameters,
        enable_rerun_server=True,
        publish_observations=enable_realtime_updates,
    )
    # Call demo_main with all the arguments
    demo_main(
        robot,
        parameters=parameters,
        visualize=visualize,
        manual_wait=manual_wait,
        output_filename=output_filename,
        navigate_home=navigate_home,
        device_id=device_id,
        verbose=verbose,
        show_intermediate_maps=show_intermediate_maps,
        show_final_map=show_final_map,
        show_paths=show_paths,
        random_goals=random_goals,
        force_explore=force_explore,
        no_manip=no_manip,
        explore_iter=explore_iter,
        write_instance_images=write_instance_images,
        parameter_file=parameter_file,
        enable_realtime_updates=enable_realtime_updates,
        reset=reset,
        save=save,
        **kwargs,
    )


def demo_main(
    robot: AbstractRobotClient,
    visualize,
    manual_wait,
    output_filename,
    navigate_home: bool = False,
    device_id: int = 0,
    verbose: bool = True,
    show_intermediate_maps: bool = False,
    show_final_map: bool = False,
    show_paths: bool = False,
    random_goals: bool = True,
    force_explore: bool = False,
    no_manip: bool = False,
    explore_iter: int = 10,
    write_instance_images: bool = False,
    parameters: Optional[Parameters] = None,
    parameter_file: str = "config/default.yaml",
    reset: bool = False,
    enable_realtime_updates: bool = False,
    save: bool = True,
    **kwargs,
):
    """
    Including only some selected arguments here.

    Args:
        show_intermediate_maps(bool): show maps as we explore
        show_final_map(bool): show the final 3d map after moving around and mapping the world
        show_paths(bool): display paths after planning
        random_goals(bool): randomly sample frontier goals instead of looking for closest
    """

    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    output_pkl_filename = output_filename + "_" + formatted_datetime + ".pkl"

    if parameters is None:
        print("- Load parameters")
        parameters = get_parameters(parameter_file)
        print(parameters)

    click.echo("Will connect to a Stretch robot and collect a short trajectory.")
    print("- Connect to Stretch")

    if explore_iter >= 0:
        parameters["exploration_steps"] = explore_iter
    object_to_find, location_to_place = parameters.get_task_goals()

    if write_instance_images or object_to_find is not None:
        print("- Create semantic sensor based on detic")
        semantic_sensor = create_semantic_sensor(
            parameters=parameters,
            device_id=device_id,
            verbose=verbose,
        )
    else:
        semantic_sensor = None

    print("- Start robot agent with data collection")
    demo = RobotAgent(
        robot, parameters, semantic_sensor, enable_realtime_updates=enable_realtime_updates
    )
    demo.start(goal=object_to_find, visualize_map_at_start=show_intermediate_maps)
    if reset:
        demo.move_closed_loop([0, 0, 0], max_time=60.0)

    if object_to_find is not None:
        print(f"\nSearch for {object_to_find} and {location_to_place}")
        matches = demo.get_found_instances_by_class(object_to_find)
        print(f"Currently {len(matches)} matches for {object_to_find}.")
    else:
        matches = []

    # Rotate in place
    if parameters["agent"]["in_place_rotation_steps"] > 0:
        demo.rotate_in_place(
            steps=parameters["agent"]["in_place_rotation_steps"],
            visualize=show_intermediate_maps,
        )

    # Run the actual procedure
    try:
        if len(matches) == 0 or force_explore:
            if object_to_find is not None:
                print(f"Exploring for {object_to_find}...")
            demo.run_exploration(
                manual_wait,
                explore_iter=parameters["exploration_steps"],
                task_goal=object_to_find,
                random_goals=False,
                go_home_at_end=navigate_home,
                visualize=show_intermediate_maps,
            )
        print("Done collecting data.")
        if object_to_find is not None:
            matches = demo.get_found_instances_by_class(object_to_find)
            print("-> Found", len(matches), f"instances of class {object_to_find}.")

    except Exception as e:
        raise (e)
    finally:

        # Stop updating the map
        demo.stop_realtime_updates()

        if show_final_map:
            pc_xyz, pc_rgb = demo.voxel_map.show()
        else:
            pc_xyz, pc_rgb = demo.voxel_map.get_xyz_rgb()

        if pc_rgb is None:
            return

        # Create pointcloud and write it out
        if len(output_pkl_filename) > 0:
            print(f"Write pkl to {output_pkl_filename}...")
            demo.voxel_map.write_to_pickle(output_pkl_filename)

        if save:
            demo.save_map()

        if write_instance_images:
            demo.save_instance_images(".")

        demo.go_home()
        robot.stop()


if __name__ == "__main__":
    main()
