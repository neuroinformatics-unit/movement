import numpy as np
import pytest
import xarray as xr
from matplotlib import pyplot as plt

from movement.plot import trajectory

plt.switch_backend("Agg")  # to avoid pop-up window


@pytest.fixture
def single_cross():
    """Sample data for plot testing.

    Data has five keypoints for one cross shaped mouse that is centered
    around the origin and moves forwards along the positive y axis with
    steps of 1.

    Keypoint starting position (x, y):
    - left (-1, 0)
    - centre (0, 0)
    - right (1, 0)
    - snout (0, 1)
    - tail (0, -1)

    """
    time_steps = 4
    individuals = ["individual_0"]
    keypoints = ["left", "centre", "right", "snout", "tail"]
    space = ["x", "y"]
    positions = {
        "left": {"x": -1, "y": np.arange(time_steps)},
        "centre": {"x": 0, "y": np.arange(time_steps)},
        "right": {"x": 1, "y": np.arange(time_steps)},
        "snout": {"x": 0, "y": np.arange(time_steps) + 1},
        "tail": {"x": 0, "y": np.arange(time_steps) - 1},
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

    ds = xr.DataArray(
        position_data,
        name="position",
        dims=["time", "space", "keypoints", "individuals"],
        coords={
            "time": time,
            "space": space,
            "keypoints": keypoints,
            "individuals": individuals,
        },
    )
    return ds


@pytest.fixture
def two_crosses(single_cross):
    """Return a position array with two cross-shaped mice.

    The 0-th mouse is moving forwards along the positive y axis, i.e. same as
    in sample_data_one_cross, the 1-st mouse is moving in the opposite
    direction, i.e. with it's snout towards the negative side of the y axis.
    """
    da_id1 = single_cross.copy()
    da_id1.loc[dict(space="y")] = da_id1.sel(space="y") * -1
    da_id1 = da_id1.assign_coords(individuals=["individual_1"])
    return xr.concat([single_cross.copy(), da_id1], "individuals")


@pytest.mark.parametrize(
    ["image", "selection", "expected_data"],
    [
        pytest.param(
            True,
            {"keypoints": ["left", "right"]},
            np.array([[0, 0], [0, 1], [0, 2], [0, 3]], dtype=float),
            id="left-right + image",
        ),
        pytest.param(
            True,
            {"keypoints": ["snout", "tail"]},
            np.array([[0, 0], [0, 1], [0, 2], [0, 3]], dtype=float),
            id="snout-tail",
        ),
        pytest.param(
            False,
            {"keypoints": "centre"},
            np.array([[0, 0], [0, 1], [0, 2], [0, 3]], dtype=float),
            id="centre",
        ),
        pytest.param(
            False,
            {},
            np.array([[0, 0], [0, 1], [0, 2], [0, 3]], dtype=float),
            id="no specified keypoints or individuals",
        ),
        pytest.param(
            False,
            {"individual": "individual_0"},
            np.array([[0, 0], [0, 1], [0, 2], [0, 3]], dtype=float),
            id="only individual specified",
        ),
        pytest.param(
            False,
            {"keypoints": ["centre", "snout"]},
            np.array([[0, 0.5], [0, 1.5], [0, 2.5], [0, 3.5]], dtype=float),
            id="centre-snout",
        ),
        pytest.param(
            True,
            {"keypoints": ["centre", "snout"]},
            np.array([[0, 0.5], [0, 1.5], [0, 2.5], [0, 3.5]], dtype=float),
            id="centre-snout + image",
        ),
    ],
)
def test_trajectory(single_cross, image, selection, expected_data):
    """Test trajectory plot."""
    da = single_cross
    _, ax = plt.subplots()
    if image:
        ax.imshow(np.zeros((10, 10)))
    _, ax = trajectory(da, ax=ax, **selection)
    output_data = ax.collections[0].get_offsets().data
    np.testing.assert_array_almost_equal(output_data, expected_data)


@pytest.mark.parametrize(
    ["selection", "dropped_dim"],
    [
        pytest.param(
            {"keypoints": "centre"},
            "keypoints",
            id="no_keypoints",
        ),
        pytest.param(
            {"keypoints": ["left", "right"]},
            "individuals",
            id="no_individuals",
        ),
        pytest.param(
            {"keypoints": "centre"},
            ["individuals", "keypoints"],
            id="only_time_space",
        ),
    ],
)
def test_trajectory_dropped_dim(single_cross, selection, dropped_dim):
    """Test trajectory plot without keypoints and/or individuals dimensions."""
    da = single_cross.sel(**selection)
    if "keypoints" in dropped_dim:
        da = da.drop("keypoints").squeeze()
    if "individuals" in dropped_dim:
        da = da.drop("individuals").squeeze()

    _, ax = trajectory(da)

    output_data = ax.collections[0].get_offsets().data
    expected_data = np.array([[0, 0], [0, 1], [0, 2], [0, 3]], dtype=float)
    np.testing.assert_array_almost_equal(output_data, expected_data)


@pytest.mark.parametrize(
    ["selection", "expected_data"],
    [
        pytest.param(
            {},
            np.array([[0, 0], [0, 1], [0, 2], [0, 3]], dtype=float),
            id="default",
        ),
        pytest.param(
            {"individual": "individual_0"},
            np.array([[0, 0], [0, 1], [0, 2], [0, 3]], dtype=float),
            id="individual_0",
        ),
        pytest.param(
            {"keypoints": ["snout", "tail"]},
            np.array([[0, 0], [0, 1], [0, 2], [0, 3]], dtype=float),
            id="snout-tail",
        ),
        pytest.param(
            {"individual": "individual_0", "keypoints": ["tail"]},
            np.array([[0, -1], [0, 0], [0, 1], [0, 2]], dtype=float),
            id="tail individual_0",
        ),
        pytest.param(
            {"individual": "individual_1", "keypoints": ["tail"]},
            np.array([[0, 1], [0, 0], [0, -1], [0, -2]], dtype=float),
            id="tail individual_1",
        ),
    ],
)
def test_trajectory_two_crosses(two_crosses, selection, expected_data):
    da = two_crosses
    _, ax = trajectory(da, **selection)
    output_data = ax.collections[0].get_offsets().data
    np.testing.assert_array_almost_equal(output_data, expected_data)


def test_trajectory_multiple_individuals(two_crosses):
    """Test trajectory plot with two individuals selected."""
    with pytest.raises(
        ValueError, match="Only one individual can be selected."
    ):
        trajectory(two_crosses, individual=["individual_0", "individual_1"])
