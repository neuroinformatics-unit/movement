import numpy as np

from movement.io.load_keypoints import load_keypoints


def test_load_keypoints_with_fps():
    keypoints, times = load_keypoints("sample_keypoints.csv", fps=30)
    assert len(times) == len(keypoints)
    assert np.isclose(times[1] - times[0], 1 / 30)


def test_load_keypoints_with_timestamps():
    timestamps = [0, 0.033, 0.067, 0.1]
    keypoints, times = load_keypoints(
        "sample_keypoints.csv", timestamps=timestamps
    )
    assert np.array_equal(times, np.array(timestamps))


def test_load_keypoints_invalid_timestamp_length():
    timestamps = [0, 0.033]  # Shorter than keypoints
    try:
        load_keypoints("sample_keypoints.csv", timestamps=timestamps)
    except ValueError as e:
        assert "Length of timestamps must match" in str(e)
