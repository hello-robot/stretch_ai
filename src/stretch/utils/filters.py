from typing import Any, List, Tuple
from threading import Lock
import time

class TemporalFilter:
    def __init__(self, observation_history_window_size_secs: float = 10.0, observation_history_window_size_n: int = 10):
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
        while len(self._observation_history) > 0 and (
            time.time() - self._observation_history[0][1]
        ) > self.observation_history_window_size_secs:
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

    def get_observations_from_history(
        self, acquire_lock: bool = True
    ) -> List[Any]:
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
