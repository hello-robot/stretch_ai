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

from stretch.utils.loop_stats import LoopStats


def test_loop_mark_start():
    """Verify that loop start time works properly"""
    test_loop_name = "TestLoop"
    test_loop_rate = 100
    test_stats = LoopStats(test_loop_name, test_loop_rate)

    test_stats.mark_start()
    time_ = time.perf_counter_ns()
    assert np.isclose(test_stats.ts_loop_start, time_)


def test_loop_rate_avg():
    """Verify that loop rate averages out correctly after few iterations"""
    test_loop_name = "TestLoopAvg"
    target_freq = 75.0
    test_stats = LoopStats(loop_name=test_loop_name, target_loop_rate=target_freq)
    iterations = 50  # Number of iterations to check average loop rate
    for i in range(iterations):
        test_stats.mark_start()
        time.sleep(1 / target_freq)
        test_stats.mark_end()

    # Average calculated over the last 100 runs, so a few dry runs to ensure fair checking
    for i in range(iterations):
        test_stats.mark_start()
        time.sleep(1 / target_freq)
        test_stats.mark_end()
        assert np.isclose(test_stats.status["avg_rate_hz"], target_freq, atol=5)


def test_loop_rate_min():
    test_loop_name = "TestLoopMin"
    test_loop_rate = 5.0
    test_stats = LoopStats(loop_name=test_loop_name, target_loop_rate=test_loop_rate)

    loop_rate_target = [3.0, 3.125, 3.25, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]

    for target_freq in loop_rate_target:
        test_stats.mark_start()
        time.sleep(1 / target_freq)
        test_stats.mark_end()

    assert np.isclose(test_stats.status["min_rate_hz"], 3.0, atol=1)


def test_loop_rate_max():
    test_loop_name = "TestLoopRateMax"
    test_loop_rate = 5.0
    test_stats = LoopStats(loop_name=test_loop_name, target_loop_rate=test_loop_rate)

    loop_rate_target = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 25.0, 25.1]

    for target_freq in loop_rate_target:
        test_stats.mark_start()
        time.sleep(1 / target_freq)
        test_stats.mark_end()

    assert np.isclose(test_stats.status["max_rate_hz"], 25.1, atol=1)


# def test_execution_time_ms():
#    test_loop_name = "TestLoopExecution"
#    test_loop_rate = 5.0
#    test_stats = LoopStats(loop_name=test_loop_name, target_loop_rate=test_loop_rate)
#
#    loop_rate_target = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
#
#    test_stats.mark_start()
#    test_stats.mark_end()
#
#    assert np.isclose(test_stats.status["execution_time_ns"], 0, atol=1e3)
#
#    for target_freq in loop_rate_target:
#        test_stats.mark_start()
#        time.sleep(1 / target_freq)
#        test_stats.mark_end()
#
#        assert np.isclose(test_stats.status["execution_time_ns"], (1 / target_freq) * 1e9, atol=1e7)
