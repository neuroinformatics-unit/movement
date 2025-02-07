import numpy as np
import pytest
import xarray as xr
from matplotlib import pyplot as plt

from movement.plot import trajectory

plt.switch_backend("Agg")  # to avoid pop-up window


@pytest.fixture
def sample_data():
    """Sample data for plot testing.

    Data has six keypoints for one individual that moves in a straight line
    along the y-axis. All keypoints have a constant x-coordinate and move in
    steps of 1 along the y-axis.

    Keypoint starting positions:
    - left1: (-1, 0)
    - right1: (1, 0)
    - left2: (-2, 0)
    - right2: (2, 0)
    - centre0: (0, 0)
    - centre1: (0, 1)

    """
    time_steps = 4
    individuals = ["individual_0"]
    keypoints = ["left1", "centre0", "right1", "left2", "centre1", "right2"]
    space = ["x", "y"]
    positions = {
        "left1": {"x": -1, "y": np.arange(time_steps)},
        "centre0": {"x": 0, "y": np.arange(time_steps)},
        "right1": {"x": 1, "y": np.arange(time_steps)},
        "left2": {"x": -2, "y": np.arange(time_steps)},
        "centre1": {"x": 0, "y": np.arange(time_steps) + 1},
        "right2": {"x": 2, "y": np.arange(time_steps)},
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


@pytest.mark.parametrize(
    ["image", "selection", "expected_data"],
    [
        pytest.param(
            True,
            {"keypoints": ["left1", "right1"]},
            np.array([[0, 0], [0, 1], [0, 2], [0, 3]], dtype=float),
            id="left1-right1 + image",
        ),
        pytest.param(
            True,
            {"keypoints": ["left2", "right2"]},
            np.array([[0, 0], [0, 1], [0, 2], [0, 3]], dtype=float),
            id="left2-right2",
        ),
        pytest.param(
            False,
            {"keypoints": "centre0"},
            np.array([[0, 0], [0, 1], [0, 2], [0, 3]], dtype=float),
            id="centre0",
        ),
        pytest.param(
            False,
            {"keypoints": "centre1"},
            np.array([[0, 1], [0, 2], [0, 3], [0, 4]], dtype=float),
            id="centre1",
        ),
        pytest.param(
            False,
            None,
            np.array(
                [
                    [0.0, 0.16666667],
                    [0.0, 1.16666667],
                    [0.0, 2.16666667],
                    [0.0, 3.16666667],
                ]
            ),
            id="no specified keypoints or individuals",
        ),
        pytest.param(
            False,
            {"individuals": "individual_0"},
            np.array(
                [
                    [0.0, 0.16666667],
                    [0.0, 1.16666667],
                    [0.0, 2.16666667],
                    [0.0, 3.16666667],
                ]
            ),
            id="only individual specified",
        ),
    ],
)
def test_trajectory(sample_data, image, selection, expected_data):
    """Test trajectory plot."""
    da = sample_data.position
    _, ax = plt.subplots()
    if image:
        ax.imshow(np.zeros((10, 10)))

    _, ax = trajectory(da, selection=selection, ax=ax)
    output_data = ax.collections[0].get_offsets().data
    np.testing.assert_array_almost_equal(output_data, expected_data)


@pytest.mark.parametrize(
    ["selection", "dropped_dim"],
    [
        pytest.param(
            {"keypoints": "centre0"},
            "keypoints",
            id="no_keypoints",
        ),
        pytest.param(
            {"keypoints": ["left1", "right1"]},
            "individuals",
            id="no_individuals",
        ),
        pytest.param(
            {"keypoints": "centre0"},
            ["individuals", "keypoints"],
            id="only_time_space",
        ),
    ],
)
def test_trajectory_dropped_dim(sample_data, selection, dropped_dim):
    """Test trajectory plot without keypoints and/or individuals dimensions."""
    position = sample_data.position.sel(**selection)
    if "keypoints" in dropped_dim:
        position = position.drop("keypoints").squeeze()
    if "individuals" in dropped_dim:
        position = position.drop("individuals").squeeze()

    _, ax = trajectory(position)

    output_data = ax.collections[0].get_offsets().data
    expected_data = np.array([[0, 0], [0, 1], [0, 2], [0, 3]], dtype=float)
    np.testing.assert_array_almost_equal(output_data, expected_data)
