"""Unit tests for the plot module."""

import numpy as np
import pytest
import xarray as xr

from movement.plot import vector


def create_sample_data(keypoints, positions):
    """Create sample data for testing."""
    time_steps = 4
    individuals = ["individual_0"]
    space = ["x", "y"]

    time = np.arange(time_steps)
    position_data = np.zeros(
        (time_steps, len(space), len(keypoints), len(individuals))
    )

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
    keypoints = ["left1", "right1", "left2", "right2", "centre0", "centre1"]
    positions = {
        "left1": {"x": -1, "y": np.arange(4)},
        "right1": {"x": 1, "y": np.arange(4)},
        "left2": {"x": -2, "y": np.arange(4)},
        "right2": {"x": 2, "y": np.arange(4)},
        "centre0": {"x": 0, "y": np.arange(4)},
        "centre1": {"x": 0, "y": np.arange(4) + 1},
    }
    return create_sample_data(keypoints, positions)


@pytest.mark.parametrize(
    ["vector_point", "expected_u", "expected_v"],
    [
        pytest.param(
            "centre0",
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            id="u = 0, v = 0",
        ),
        pytest.param(
            "centre1",
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            id="u = 0, v = 1",
        ),
        pytest.param(
            "right2",
            [2.0, 2.0, 2.0, 2.0],
            [0.0, 0.0, 0.0, 0.0],
            id="u = 2, v = 0",
        ),
        pytest.param(
            "left2",
            [-2.0, -2.0, -2.0, -2.0],
            [0.0, 0.0, 0.0, 0.0],
            id="u = -2, v = 0",
        ),
    ],
)
def test_vector(sample_data, vector_point, expected_u, expected_v):
    """Test vector plot.

    Test the vector plot for different vector points. The U and V values
    represent the horizontal (x) and vertical (y) displacement of the vector

    The reference points are "left1" and "right1".
    """
    vector_fig = vector(
        sample_data,
        reference_points=["left1", "right1"],
        vector_point=vector_point,
    )

    quiver = vector_fig.axes[0].collections[-1]

    x = quiver.X
    y = quiver.Y
    u = quiver.U
    v = quiver.V

    expected_x = np.array([0.0, 0.0, 0.0, 0.0])
    expected_y = np.array([0.0, 1.0, 2.0, 3.0])

    assert np.allclose(x, expected_x)
    assert np.allclose(y, expected_y)
    assert np.allclose(u, expected_u)
    assert np.allclose(v, expected_v)
