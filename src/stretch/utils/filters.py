# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import time
from threading import Lock
from typing import Any, List, Tuple

import numpy as np


class TemporalFilter:
    def __init__(
        self,
        observation_history_window_size_secs: float = 10.0,
        observation_history_window_size_n: int = 10,
    ):
        self._observation_history: List[Tuple[Any, float]] = []
        self._observation_history_lock = Lock()
        self.observation_history_window_size_secs = observation_history_window_size_secs
        self.observation_history_window_size_n = observation_history_window_size_n

    def prune_observation_history(self, acquire_lock: bool = True) -> None:
        """
        Prune the observation history to keep only the most recent observations. Note: the
        observation history lock must be acquired before calling this function.

        Parameters
        ----------
        acquire_lock : bool
            Whether to acquire the observation history lock before pruning the observation history.
        """

        if acquire_lock:
            self._observation_history_lock.acquire()
        # Prune the observation history to keep the `observation_history_window_size_n` most recent observations
        # within the `observation_history_window_size_secs` time window.
        while len(self._observation_history) > self.observation_history_window_size_n:
            self._observation_history.pop(0)
        while (
            len(self._observation_history) > 0
            and (time.time() - self._observation_history[0][1])
            > self.observation_history_window_size_secs
        ):
            self._observation_history.pop(0)
        if acquire_lock:
            self._observation_history_lock.release()

    def push_to_observation_history(
        self,
        observation: Any,
        timestamp: float,
        acquire_lock: bool = True,
    ) -> None:
        """
        Push a observation to the observation history. Note: the observation history lock must be
        acquired before calling this function.

        Parameters
        ----------
        observation : HumanobservationEstimate
            The observation to push to the observation history.
        timestamp : Time
            The timestamp of the observation.
        acquire_lock : bool
            Whether to acquire the observation history lock before pushing to the observation history.
        """
        if acquire_lock:
            self._observation_history_lock.acquire()
        self._observation_history.append((observation, timestamp))
        self.prune_observation_history(acquire_lock=False)
        if acquire_lock:
            self._observation_history_lock.release()

    def get_observations_from_history(self, acquire_lock: bool = True) -> List[Any]:
        """
        Get the observations from the observation history. Note: the observation history lock must be
        acquired before calling this function.

        Parameters
        ----------
        acquire_lock : bool
            Whether to acquire the observation history lock before getting observations from the observation history.

        Returns
        -------
        List[Any]
            The observations from the observation history.
        """
        if acquire_lock:
            self._observation_history_lock.acquire()
        self.prune_observation_history(acquire_lock=False)
        retval = [observation for observation, _ in self._observation_history]
        if acquire_lock:
            self._observation_history_lock.release()
        return retval

    def get_latest_observation(self) -> Any:
        """
        Get the latest observation from the observation history.

        Returns:
        --------
        Any
            The latest observation from the observation history.
        """
        observations = self.get_observations_from_history()
        if len(observations) == 0:
            return None
        return observations[-1]

    def clear_history(self) -> None:
        self._observation_history = []


class MaskTemporalFilter(TemporalFilter):
    @staticmethod
    def mask_centroid(mask: np.ndarray):
        """
        computes the centroid of a mask in image space
        """
        num_mask_pts = MaskTemporalFilter.count_mask_pixels(mask)

        if num_mask_pts == 0:
            mask_center = None
        else:
            mask_pts = np.argwhere(mask)
            mask_center = mask_pts.mean(axis=0)

        return mask_center

    @staticmethod
    def count_mask_pixels(mask: np.ndarray) -> int:
        return sum(mask.flatten())

    def push_mask_to_observation_history(
        self,
        observation: Any,
        timestamp: float,
        mask_size_threshold: int = 0,
        acquire_lock: bool = True,
    ) -> None:
        mask_size = self.count_mask_pixels(observation)
        if mask_size > mask_size_threshold:
            self.push_to_observation_history(
                observation=observation, timestamp=timestamp, acquire_lock=acquire_lock
            )

    def get_average_centroid(self) -> np.ndarray:
        observations = self.get_observations_from_history()
        if len(observations) == 0:
            # No observations to average
            return None

        centroids = [self.mask_centroid(observation) for observation in observations]
        centroids = [centroid for centroid in centroids if centroid is not None]
        if len(centroids) == 0:
            # No centroids to average
            return None

        avg_centroid = np.mean(centroids, axis=0)

        return avg_centroid

    def get_latest_centroid(self) -> np.ndarray:
        observations = self.get_observations_from_history()
        if len(observations) == 0:
            # No observations to average
            return None

        done = False
        while not done:
            latest_observation = observations[-1]
            latest_centroid = self.mask_centroid(latest_observation)
            if latest_centroid is not None:
                done = True
            else:
                observations = observations[:-1]
                if len(observations) == 0:
                    # No centroids to average
                    return None

        return latest_centroid
