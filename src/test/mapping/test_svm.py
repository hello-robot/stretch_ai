# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import numpy as np

from stretch.agent import RobotAgent
from stretch.core import Parameters
from stretch.utils.config import Config
from stretch.utils.dummy_stretch_client import DummyStretchClient

SMALL_DATA_FILE = "test/mapping/hq_small.pkl"
LARGE_DATA_FILE = "test/mapping/hq_large.pkl"
TEST_PLANNER_FILENAME = "test/mapping/planner.yaml"

SMALL_DATA_START = np.array([4.5, 1.4, 0.0])
LARGE_DATA_START = np.array([4.5, 1.4, 0.0])

queries = [
    ("cardboard box", True),
    ("photo of an elephant", False),
]

similarity_threshold = 0.05
debug = False


def _eval_svm(filename: str, start_pos: np.ndarray, possible: bool = False) -> None:

    print("==== SVM Evaluation ====")
    print(f"Loading voxel map from {filename}...")
    config = Config()
    config.merge_from_file(TEST_PLANNER_FILENAME)
    config.freeze()
    parameters = Parameters(**config)

    print("Create dummy robot and agent...")
    dummy_robot = DummyStretchClient()
    agent = RobotAgent(
        dummy_robot,
        parameters,
        semantic_sensor=None,
        voxel_map=None,
        use_instance_memory=True,
    )
    voxel_map = agent.voxel_map

    print("Reading from pkl file of raw observations...")
    frame = -1
    semantic_sensor = None
    ok = voxel_map.read_from_pickle(filename, num_frames=frame, perception=semantic_sensor)

    print(f"Reading from pkl file of raw observations... {ok=}")
    assert ok, "Failed to read from pkl file of raw observations"

    print("Evaluating SVM...")
    print("# Instances =", len(voxel_map.get_instances()))
    assert len(voxel_map.get_instances()) > 0, "No instances found in voxel map"

    assert agent.get_navigation_space() is not None, "Failed to create navigation space"
    navigation_space = agent.get_navigation_space()
    assert navigation_space is not None, "Failed to create navigation space"
    assert navigation_space.is_valid(
        start_pos, verbose=True
    ), f"Start position is not valid: {start_pos}"

    # Show the map
    if debug:
        voxel_map.show(orig=np.zeros(3), xyt=start_pos, footprint=dummy_robot.get_footprint())

    for query, expected_result in queries:
        instances = agent.get_ranked_instances(query)
        # Query the SVM - make sure we can find motion plan to a cardboard box
        score, instance_id, instance = instances[0]
        print(f"Query: {query} Score: {score} Instance ID: {instance_id}")
        assert instance_id is not None, "Failed to find instance ID"
        if expected_result:
            assert (
                score > similarity_threshold
            ), f"Failed to find instance with positive score for {query}"
        else:
            # TODO: remove debug code
            # import matplotlib.pyplot as plt
            # plt.imshow(instance.get_best_view().get_image())
            # plt.show()
            assert score < similarity_threshold, f"Found instance with positive score for {query}"

        if expected_result and possible:
            # Try motion planning to matching instances
            for i, (score, instance_id, instance) in enumerate(instances):
                if score < similarity_threshold:
                    assert False, "Failed to find instance with acceptable score"
                res = agent.plan_to_instance(instance, start_pos, verbose=False, radius_m=0.3)
                print(f"Plan to instance {i}={instance.global_id} = {res.success}")
                if res.success:
                    break
            else:
                assert False, "Failed to find a plan to any acceptable instance for {query}"

    # Plan to the frontier
    print("Plan to the frontier")
    res = agent.plan_to_frontier(start_pos)
    print(f"Plan to the frontier = {res.success}")
    assert res.success, f"Failed to plan to the frontier: {res.reason}"

    # Test deletion
    print("Test deletion")
    for query, expected_result in queries:
        if not expected_result:
            continue

        # Just get one instance
        instances = agent.get_ranked_instances(query)
        score, instance_id, instance = instances[0]

        # Delete the instance
        voxel_map.delete_instance(instance)

        # Try to find the instance again
        instances = agent.get_ranked_instances(query)
        new_score, new_instance_id, new_instance = instances[0]

        assert (
            new_score < score
        ), f"Failed to delete instance or instances were not returned in the right order; {new_score} >= {score}"
        assert (
            new_instance_id != instance_id
        ), f"Failed to delete instance; {new_instance_id} == {instance_id}"


def test_svm_small():
    _eval_svm(SMALL_DATA_FILE, SMALL_DATA_START, possible=False)


def test_svm_large():
    _eval_svm(LARGE_DATA_FILE, LARGE_DATA_START, possible=True)


if __name__ == "__main__":
    debug = True
    # test_svm_small()
    test_svm_large()
