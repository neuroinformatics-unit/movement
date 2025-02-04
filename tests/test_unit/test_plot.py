import numpy as np
import pytest
import xarray as xr

from movement.plot import vector


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


@pytest.fixture
def sample_data_quiver1():
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
        "centre": {"x": 0, "y": np.arange(time_steps) + 1},
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


@pytest.fixture
def sample_data_quiver2():
    """Sample data for plot testing.

    Data has three keypoints (left, centre, right) for one
    individual that moves in a straight line along the y-axis with a
    constant x-coordinate.

    """
    time_steps = 4
    individuals = ["individual_0"]
    keypoints = ["left1", "right1", "left2", "right2"]
    space = ["x", "y"]
    positions = {
        "left1": {"x": -1, "y": np.arange(time_steps)},
        "right1": {"x": 1, "y": np.arange(time_steps)},
        "left2": {"x": -1, "y": np.arange(time_steps)},
        "right2": {"x": 1, "y": np.arange(time_steps)},
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


def test_vector_no_quiver(sample_data):
    """Test midpoint between left and right keypoints."""
    vector_fig = vector(
        sample_data,
        reference_points=["left", "right"],
        vector_point="centre",
    )

    quiver = vector_fig.axes[0].collections[-1]

    # Extract the X, Y, U, V data
    x = quiver.X
    y = quiver.Y
    u = quiver.U
    v = quiver.V

    expected_x = np.array([0.0, 0.0, 0.0, 0.0])
    expected_y = np.array([0.0, 1.0, 2.0, 3.0])
    expected_u = np.array([0.0, 0.0, 0.0, 0.0])
    expected_v = np.array([0.0, 0.0, 0.0, 0.0])

    assert np.allclose(x, expected_x)
    assert np.allclose(y, expected_y)
    assert np.allclose(u, expected_u)
    assert np.allclose(v, expected_v)


def test_vector_quiver2(sample_data_quiver2):
    """Test midpoint between left and right keypoints."""
    vector_fig = vector(
        sample_data_quiver2,
        reference_points=["left1", "right1"],
        vector_point="right2",
    )

    quiver = vector_fig.axes[0].collections[-1]

    # Extract the X, Y, U, V data
    x = quiver.X
    y = quiver.Y
    u = quiver.U
    v = quiver.V

    expected_x = np.array([0.0, 0.0, 0.0, 0.0])
    expected_y = np.array([0.0, 1.0, 2.0, 3.0])
    expected_u = np.array([1.0, 1.0, 1.0, 1.0])
    expected_v = np.array([0.0, 0.0, 0.0, 0.0])

    assert np.allclose(x, expected_x)
    assert np.allclose(y, expected_y)
    assert np.allclose(u, expected_u)
    assert np.allclose(v, expected_v)


def test_vector_quiver1(sample_data_quiver1):
    """Test midpoint between left and right keypoints."""
    vector_fig = vector(
        sample_data_quiver1,
        reference_points=["left", "right"],
        vector_point="centre",
    )

    quiver = vector_fig.axes[0].collections[-1]

    # Extract the X, Y, U, V data
    x = quiver.X
    y = quiver.Y
    u = quiver.U
    v = quiver.V

    expected_x = np.array([0.0, 0.0, 0.0, 0.0])
    expected_y = np.array([0.0, 1.0, 2.0, 3.0])
    expected_u = np.array([0.0, 0.0, 0.0, 0.0])
    expected_v = np.array([1.0, 1.0, 1.0, 1.0])

    assert np.allclose(x, expected_x)
    assert np.allclose(y, expected_y)
    assert np.allclose(u, expected_u)
    assert np.allclose(v, expected_v)
