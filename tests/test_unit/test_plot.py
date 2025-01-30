import numpy as np
import pytest
import xarray as xr
from matplotlib import pyplot as plt

from movement.plot import trajectory


@pytest.fixture
def sample_data():
    """Sample data for plot testing.

    Data has three keypoints (left, centre, right) for one
    individual that moves in a straight line along the y-axis with a
    constant x-coordinate.

    """
    time_steps = 4
    individuals = ["individual_0"]
    keypoints = ["left", "centre", "right"]
    space = ["x", "y"]
    positions = {
        "left": {"x": -1, "y": np.arange(time_steps)},
        "centre": {"x": 0, "y": np.arange(time_steps)},
        "right": {"x": 1, "y": np.arange(time_steps)},
    }

    time = np.arange(time_steps)
    position_data = np.zeros(
        (time_steps, len(space), len(keypoints), len(individuals))
    )

    # Create x and y coordinates arrays
    x_coords = np.array([positions[key]["x"] for key in keypoints])
    y_coords = np.array([positions[key]["y"] for key in keypoints])

    for i, _ in enumerate(keypoints):
        position_data[:, 0, i, 0] = x_coords[i]  # x-coordinates
        position_data[:, 1, i, 0] = y_coords[i]  # y-coordinates

    confidence_data = np.full(
        (time_steps, len(keypoints), len(individuals)), 0.90
    )

    ds = xr.Dataset(
        {
            "position": (
                ["time", "space", "keypoints", "individuals"],
                position_data,
            ),
            "confidence": (
                ["time", "keypoints", "individuals"],
                confidence_data,
            ),
        },
        coords={
            "time": time,
            "space": space,
            "keypoints": keypoints,
            "individuals": individuals,
        },
    )
    return ds


def test_trajectory(sample_data):
    """Test midpoint between left and right keypoints."""
    plt.switch_backend("Agg")  # to avoid pop-up window
    fig_centre = trajectory(sample_data, keypoint="centre")
    fig_left_right_midpoint = trajectory(
        sample_data, keypoint=["left", "right"]
    )

    expected_data = np.array([[0, 0], [0, 1], [0, 2], [0, 3]])

    # Retrieve data points from figures
    ax_centre = fig_centre.axes[0]
    centre_data = ax_centre.collections[0].get_offsets().data

    ax_left_right = fig_left_right_midpoint.axes[0]
    left_right_data = ax_left_right.collections[0].get_offsets().data

    np.testing.assert_array_almost_equal(centre_data, left_right_data)
    np.testing.assert_array_almost_equal(centre_data, expected_data)
    np.testing.assert_array_almost_equal(left_right_data, expected_data)


def test_trajectory_with_frame(sample_data, tmp_path):
    """Test plot trajectory with frame."""
    frame_path = tmp_path / "frame.png"
    fig, ax = plt.subplots()
    ax.imshow(np.zeros((10, 10)))
    fig.savefig(frame_path)

    fig_centre = trajectory(
        sample_data, keypoint="centre", frame_path=frame_path
    )
    fig_left_right_midpoint = trajectory(
        sample_data, keypoint=["left", "right"], frame_path=frame_path
    )

    # Retrieve data points from figures
    ax_centre = fig_centre.axes[0]
    centre_data = ax_centre.collections[0].get_offsets().data

    ax_left_right = fig_left_right_midpoint.axes[0]
    left_right_data = ax_left_right.collections[0].get_offsets().data

    expected_data = np.array([[0, 0], [0, 1], [0, 2], [0, 3]])
    np.testing.assert_array_almost_equal(centre_data, left_right_data)
    np.testing.assert_array_almost_equal(centre_data, expected_data)
    np.testing.assert_array_almost_equal(left_right_data, expected_data)
