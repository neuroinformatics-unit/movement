import numpy as np
import pytest
from matplotlib import pyplot as plt

from movement.plots.trajectory import plot_centroid_trajectory

plt.switch_backend("Agg")  # to avoid pop-up window


@pytest.mark.parametrize(
    ["image", "selection", "expected_data"],
    [
        pytest.param(
            True,
            {"keypoint": ["left", "right"]},
            np.array([[0, 0], [0, 1], [0, 2], [0, 3]], dtype=float),
            id="left-right + image",
        ),
        pytest.param(
            False,
            {"keypoint": ["snout", "tail"]},
            np.array([[0, 0], [0, 1], [0, 2], [0, 3]], dtype=float),
            id="snout-tail",
        ),
        pytest.param(
            False,
            {"keypoint": "centre"},
            np.array([[0, 0], [0, 1], [0, 2], [0, 3]], dtype=float),
            id="centre",
        ),
        pytest.param(
            False,
            {},
            np.array([[0, 0], [0, 1], [0, 2], [0, 3]], dtype=float),
            id="no specified keypoint or individual",
        ),
        pytest.param(
            False,
            {"individual": "id_0"},
            np.array([[0, 0], [0, 1], [0, 2], [0, 3]], dtype=float),
            id="only individual specified",
        ),
        pytest.param(
            False,
            {"keypoint": ["centre", "snout"]},
            np.array([[0, 0.5], [0, 1.5], [0, 2.5], [0, 3.5]], dtype=float),
            id="centre-snout",
        ),
        pytest.param(
            True,
            {"keypoint": ["centre", "snout"]},
            np.array([[0, 0.5], [0, 1.5], [0, 2.5], [0, 3.5]], dtype=float),
            id="centre-snout + image",
        ),
    ],
)
def test_trajectory_plot(one_individual, image, selection, expected_data):
    """Test trajectory plot."""
    da = one_individual
    _, ax = plt.subplots()
    if image:
        ax.imshow(np.zeros((10, 10)))
    _, ax = plot_centroid_trajectory(da, ax=ax, **selection)
    output_data = ax.collections[0].get_offsets().data
    np.testing.assert_array_almost_equal(output_data, expected_data)


@pytest.mark.parametrize(
    ["selection"],
    [
        pytest.param(
            {"keypoint": "centre"},
            id="no_keypoint",
        ),
        pytest.param(
            {"individual": "id_0"},
            id="no_individual",
        ),
        pytest.param(
            {"keypoint": "centre", "individual": "id_0"},
            id="only_time_space",
        ),
    ],
)
def test_trajectory_dropped_dim(two_individual, selection):
    """Test trajectory plot without keypoint and/or individual dimensions.

    When only one coordinate is selected per dimension, that dimension will
    be squeezed out of the data array.
    """
    da = two_individual.sel(**selection).squeeze()
    _, ax = plot_centroid_trajectory(da)
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
            {"individual": "id_0"},
            np.array([[0, 0], [0, 1], [0, 2], [0, 3]], dtype=float),
            id="id_0",
        ),
        pytest.param(
            {"keypoint": ["snout", "tail"]},
            np.array([[0, 0], [0, 1], [0, 2], [0, 3]], dtype=float),
            id="snout-tail",
        ),
        pytest.param(
            {"individual": "id_0", "keypoint": ["tail"]},
            np.array([[0, -1], [0, 0], [0, 1], [0, 2]], dtype=float),
            id="tail id_0",
        ),
        pytest.param(
            {"individual": "id_1", "keypoint": ["tail"]},
            np.array([[0, 1], [0, 0], [0, -1], [0, -2]], dtype=float),
            id="tail id_1",
        ),
    ],
)
def test_trajectory_two_crosses(two_individual, selection, expected_data):
    da = two_individual
    _, ax = plot_centroid_trajectory(da, **selection)
    output_data = ax.collections[0].get_offsets().data
    np.testing.assert_array_almost_equal(output_data, expected_data)


def test_trajectory_multiple_individual(two_individual):
    """Test trajectory plot with two individual selected."""
    with pytest.raises(
        ValueError, match="Only one individual can be selected."
    ):
        plot_centroid_trajectory(two_individual, individual=["id_0", "id_1"])
