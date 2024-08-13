# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import time

import numpy as np

from stretch.utils.filters import MaskTemporalFilter, TemporalFilter


def test_temporal_filter(debug_print: bool = False):
    n_observations = 10
    observation_history_window_size_secs = 0.5
    observation_history_window_size_n = n_observations
    temporal_filter = TemporalFilter(
        observation_history_window_size_secs=observation_history_window_size_secs,
        observation_history_window_size_n=observation_history_window_size_n,
    )

    # Push `n_observations` observations to the observation history.
    start_time = time.time()
    for i in range(n_observations):
        timestamp = start_time + (i * 0.1)
        temporal_filter.push_to_observation_history(
            observation=[i], timestamp=timestamp, acquire_lock=False
        )

    assert len(temporal_filter.get_observations_from_history()) == n_observations

    if debug_print:
        for _ in range(2 * n_observations):
            time.sleep(0.1)
            print(temporal_filter.get_observations_from_history())

        assert len(temporal_filter.get_observations_from_history()) == 0
    else:
        print("Sleeping to empty queue for 1.5 * 0.1 * n_observations seconds...")
        time.sleep(1.5 * 0.1 * n_observations)
        assert len(temporal_filter.get_observations_from_history()) == 0


def test_mask_temporal_filter():
    n_observations = 10
    obs_shape = (10,)

    mask_temporal_filter = MaskTemporalFilter(
        observation_history_window_size_secs=1.0,
        observation_history_window_size_n=10,
    )

    for _ in range(n_observations):
        obs = np.random.rand(*obs_shape)
        mask_temporal_filter.push_to_observation_history(
            obs, timestamp=time.time(), acquire_lock=False
        )

    assert len(mask_temporal_filter.get_observations_from_history()) == n_observations


def test_mask_temporal_filter_static():
    test_mask = np.zeros([100, 100])
    test_mask[:50, :50] = 1
    centroid = MaskTemporalFilter.mask_centroid(test_mask)
    assert len(centroid) == 2

    mask_temporal_filter = MaskTemporalFilter(
        observation_history_window_size_secs=1.0,
        observation_history_window_size_n=10,
    )

    for _ in range(10):
        mask_temporal_filter.push_to_observation_history(
            test_mask, timestamp=time.time(), acquire_lock=False
        )

    centroid = mask_temporal_filter.get_latest_centroid()

    assert len(centroid) == 2


if __name__ == "__main__":
    test_temporal_filter()
    test_mask_temporal_filter()
    test_mask_temporal_filter_static()
