import numpy as np
import pytest
import xarray as xr
from matplotlib import pyplot as plt

from movement.plots.vector import plot_vector

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

    The left and right keypoints are not mirrored for individual_1, so this
    mouse is moving flipped around on it's back.
    """
    da_id1 = single_cross.copy()
    da_id1.loc[dict(space="y")] = da_id1.sel(space="y") * -1
    da_id1 = da_id1.assign_coords(individuals=["individual_1"])
    return xr.concat([single_cross.copy(), da_id1], "individuals")


@pytest.mark.parametrize(
    ["image", "selection", "expected_U", "expected_V"],
    [
        pytest.param(
            False,
            {},
            np.array([-1.0, -1.0, -1.0, -1.0]),
            np.array([0.0, 0.0, 0.0, 0.0]),
            id="no keypoints selected",
        ),
        pytest.param(
            False,
            {
                "reference_keypoints": ["left", "right"],
                "vector_keypoints": "centre",
            },
            np.array([0.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0, 0.0]),
            id="no vector",
        ),
        pytest.param(
            False,
            {
                "vector_keypoints": "centre",
            },
            np.array([0.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0, 0.0]),
            id="no reference keypoints",
        ),
        pytest.param(
            False,
            {
                "reference_keypoints": "left",
            },
            np.array([0.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0, 0.0]),
            id="no vector keypoints (default to first, i.e. left)",
        ),
        pytest.param(
            False,
            {
                "vector_keypoints": ["tail", "snout"],
            },
            np.array([0.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0, 0.0]),
            id="multiple vector keypoints",
        ),
        pytest.param(
            False,
            {
                "reference_keypoints": "centre",
                "vector_keypoints": "snout",
            },
            np.array([0.0, 0.0, 0.0, 0.0]),
            np.array([1.0, 1.0, 1.0, 1.0]),
            id="centre to snout",
        ),
        pytest.param(
            False,
            {
                "reference_keypoints": "snout",
                "vector_keypoints": "centre",
            },
            np.array([0.0, 0.0, 0.0, 0.0]),
            np.array([-1.0, -1.0, -1.0, -1.0]),
            id="snout to centre",
        ),
        pytest.param(
            True,
            {
                "reference_keypoints": "snout",
                "vector_keypoints": "centre",
            },
            np.array([0.0, 0.0, 0.0, 0.0]),
            np.array([-1.0, -1.0, -1.0, -1.0]),
            id="snout to centre + image",
        ),
        pytest.param(
            False,
            {
                "reference_keypoints": "centre",
                "vector_keypoints": "left",
            },
            np.array([-1.0, -1.0, -1.0, -1.0]),
            np.array([0.0, 0.0, 0.0, 0.0]),
            id="centre to left",
        ),
        pytest.param(
            False,
            {
                "reference_keypoints": "centre",
                "vector_keypoints": "right",
            },
            np.array([1.0, 1.0, 1.0, 1.0]),
            np.array([0.0, 0.0, 0.0, 0.0]),
            id="centre to right",
        ),
    ],
)
def test_vector_plot(single_cross, image, selection, expected_U, expected_V):
    """Test vector plot."""
    da = single_cross
    _, ax = plt.subplots()
    if image:
        ax.imshow(np.zeros((10, 10)))
    _, ax = plot_vector(da, ax=ax, **selection)
    U = ax.collections[-1].U
    V = ax.collections[-1].V
    np.testing.assert_array_almost_equal(U, expected_U)
    np.testing.assert_array_almost_equal(V, expected_V)


@pytest.mark.parametrize(
    ["selection", "dropped_dim"],
    [
        pytest.param(
            {"keypoints": "left"},
            "keypoints",
            id="no_keypoints",
        ),
        pytest.param(
            {"keypoints": "left"},
            ["individuals", "keypoints"],
            id="only_time_space",
        ),
    ],
)
def test_vector_no_keypoints(single_cross, selection, dropped_dim):
    """Test vector plot without keypoints dimensions."""
    da = single_cross.sel(**selection)
    if "keypoints" in dropped_dim:
        da = da.drop_vars("keypoints").squeeze()
    if "individuals" in dropped_dim:
        da = da.drop_vars("individuals").squeeze()

    with pytest.raises(
        ValueError,
        match="DataArray must have 'keypoints' dimension to plot vectors.",
    ):
        plot_vector(da)


def test_vector_no_individuals(single_cross):
    """Test vector plot without individuals dimension."""
    selection = {"keypoints": ["left", "right"]}
    da = single_cross.sel(**selection)
    da = da.drop_vars("individuals").squeeze()
    _, ax = plot_vector(da)

    U = ax.collections[-1].U
    V = ax.collections[-1].V

    expected_U = np.array([-1.0, -1.0, -1.0, -1.0])
    expected_V = np.array([0.0, 0.0, 0.0, 0.0])

    np.testing.assert_array_almost_equal(U, expected_U)
    np.testing.assert_array_almost_equal(V, expected_V)


def test_vector_multiple_individuals(two_crosses):
    """Test vector plot with two individuals selected."""
    with pytest.raises(
        ValueError, match="Only one individual can be selected."
    ):
        plot_vector(two_crosses, individual=["individual_0", "individual_1"])
