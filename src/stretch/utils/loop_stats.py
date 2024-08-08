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


class LoopStats:
    """Collects rate statistics for loops"""

    def __init__(self, loop_name, target_loop_rate=float("inf")):
        self.loop_name = loop_name
        self.target_loop_rate = target_loop_rate
        if self.target_loop_rate == 0.0:
            raise ValueError("the target loop rate must be greater than zero")
        self.ts_loop_start = None
        self.ts_loop_end = None
        self.last_ts_loop_start = None
        self.status = {
            "execution_time_ns": None,
            "curr_rate_hz": 0,
            "avg_rate_hz": 0,
            "supportable_rate_hz": 0,
            "min_rate_hz": float("inf"),
            "max_rate_hz": float("-inf"),
            "std_rate_hz": 0,
            "missed_loops": 0,
            "num_loops": 0,
        }
        self.curr_rate_history = []
        self.supportable_rate_history = []
        self.n_history = 100
        self.sleep_time_s = 0.0
        self.ts_0 = time.time()

    def pretty_print(self):
        print(f"--------- LoopStats {self.loop_name} -----------")
        print(f"Target rate (Hz): {self.target_loop_rate:.2f}")
        print(f"Current rate (Hz): {self.status['avg_rate_hz']:.2f}")
        print(f"Standard deviation of rate history (Hz): {self.status['std_rate_hz']:.2f}")
        print(f"Min rate (Hz): {self.status['min_rate_hz']:.2f}")
        print(f"Max rate (Hz): {self.status['max_rate_hz']:.2f}")
        print(f"Supportable rate (Hz): {self.status['supportable_rate_hz']:.2f}")
        print(f"Missed / Total: {self.status['missed_loops']} out of {self.status['num_loops']}")

    def mark_start(self):
        self.status["num_loops"] += 1
        self.ts_loop_start = time.perf_counter_ns()

        if self.last_ts_loop_start is None:  # Wait until have sufficient data
            self.last_ts_loop_start = time.perf_counter_ns()
            return

        # Calculate current/min/max loop rate
        self.status["curr_rate_hz"] = 1.0 / ((self.ts_loop_start - self.last_ts_loop_start) * 1e-9)
        self.status["min_rate_hz"] = min(self.status["curr_rate_hz"], self.status["min_rate_hz"])
        self.status["max_rate_hz"] = max(self.status["curr_rate_hz"], self.status["max_rate_hz"])

        # Calculate average and supportable loop rate
        if len(self.curr_rate_history) >= self.n_history:
            self.curr_rate_history.pop(0)
        self.curr_rate_history.append(self.status["curr_rate_hz"])
        self.status["avg_rate_hz"] = np.mean(self.curr_rate_history)
        self.status["std_rate_hz"] = np.std(self.curr_rate_history)
        if self.status["execution_time_ns"] is None:
            self.last_ts_loop_start = time.perf_counter_ns()
            return
        if len(self.supportable_rate_history) >= self.n_history:
            self.supportable_rate_history.pop(0)
        self.supportable_rate_history.append(1.0 / (self.status["execution_time_ns"] * 1e-9))
        self.status["supportable_rate_hz"] = np.mean(self.supportable_rate_history)

        # Calculate sleep time to achieve desired loop rate
        self.sleep_time_s = (1 / self.target_loop_rate) - (self.status["execution_time_ns"] * 1e-9)
        if (
            self.sleep_time_s < 0.0 and time.time() - self.ts_0 > 5.0
        ):  # Allow 5s for timing to stabilize on startup
            self.status["missed_loops"] += 1

        self.last_ts_loop_start = time.perf_counter_ns()

    def mark_end(self):
        # First two cycles initialize vars / log
        if self.ts_loop_start is None:
            return
        self.ts_loop_end = time.perf_counter_ns()
        self.status["execution_time_ns"] = self.ts_loop_end - self.last_ts_loop_start

    def generate_rate_histogram(self, save=None):
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)
        fig.suptitle("Distribution of loop rate (Hz). Target of %.2f " % self.target_loop_rate)
        axs.hist(
            x=self.curr_rate_history,
            bins="auto",
            color="#0504aa",
            alpha=0.7,
            rwidth=0.85,
        )
        plt.show() if save is None else plt.savefig(save)

    def get_loop_sleep_time(self):
        """
        Returns:
            float: Time to sleep for to hit target loop rate
        """
        return max(0.0, self.sleep_time_s)

    def sleep(self):
        """
        Sleeps for the remaining time to hit target loop rate
        """
        time.sleep(self.get_loop_sleep_time())
