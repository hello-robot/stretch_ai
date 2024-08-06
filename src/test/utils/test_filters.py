from stretch.utils.filters import TemporalFilter
import time

def test_temporal_filter(debug_print: bool=False):
    n_observations = 10
    observation_history_window_size_secs = 0.5
    observation_history_window_size_n = 10
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

if __name__ == "__main__":
    test_temporal_filter()