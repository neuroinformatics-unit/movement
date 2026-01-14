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
            {"keypoints": ["left", "right"]},
            np.array([[0, 0], [0, 1], [0, 2], [0, 3]], dtype=float),
            id="left-right + image",
        ),
        pytest.param(
            False,
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
            {"individual": "id_0"},
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
            id="no_keypoints",
        ),
        pytest.param(
            {"individual": "id_0"},
            id="no_individuals",
        ),
        pytest.param(
            {"keypoint": "centre", "individual": "id_0"},
            id="only_time_space",
        ),
    ],
)
def test_trajectory_dropped_dim(two_individuals, selection):
    """Test trajectory plot without keypoints and/or individuals dimensions.

    When only one coordinate is selected per dimension, that dimension will
    be squeezed out of the data array.
    """
    da = two_individuals.sel(**selection).squeeze()
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
            {"keypoints": ["snout", "tail"]},
            np.array([[0, 0], [0, 1], [0, 2], [0, 3]], dtype=float),
            id="snout-tail",
        ),
        pytest.param(
            {"individual": "id_0", "keypoints": ["tail"]},
            np.array([[0, -1], [0, 0], [0, 1], [0, 2]], dtype=float),
            id="tail id_0",
        ),
        pytest.param(
            {"individual": "id_1", "keypoints": ["tail"]},
            np.array([[0, 1], [0, 0], [0, -1], [0, -2]], dtype=float),
            id="tail id_1",
        ),
    ],
)
def test_trajectory_two_crosses(two_individuals, selection, expected_data):
    da = two_individuals
    _, ax = plot_centroid_trajectory(da, **selection)
    output_data = ax.collections[0].get_offsets().data
    np.testing.assert_array_almost_equal(output_data, expected_data)


def test_trajectory_multiple_individuals(two_individuals):
    """Test trajectory plot with two individuals selected."""
    with pytest.raises(
        ValueError, match="Only one individual can be selected."
    ):
        plot_centroid_trajectory(two_individuals, individual=["id_0", "id_1"])
