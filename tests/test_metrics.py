import numpy as np
import pytest

from movement.metrics import trajectory_complexity


def test_trajectory_complexity():
    # Simple straight-line trajectory
    x = np.array([0, 5])
    y = np.array([0, 0])
    assert trajectory_complexity(x, y) == 1.0  # Perfect straight line

    # More complex trajectory
    x = np.array([0, 1, 2, 3])
    y = np.array([0, 2, 1, 3])
    assert 0 < trajectory_complexity(x, y) < 1  # Curved path

    # Edge case: single point (should raise an error)
    with pytest.raises(ValueError):
        trajectory_complexity([0], [0])


def test_time_coordinate_loading():
    keypoints, times = load_keypoints("sample_data.csv")
    assert times is not None, "Time coordinates should be loaded"
    assert len(times) == len(keypoints), (
        "Time and keypoints should have the same length"
    )
