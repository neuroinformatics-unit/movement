from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr
from matplotlib.collections import QuadMesh
from numpy.random import RandomState

from movement.plot import occupancy_histogram


def get_histogram_binning_data(fig: plt.Figure) -> list[QuadMesh]:
    """Fetch 2D array data from a histogram plot."""
    return [
        qm for qm in fig.axes[0].get_children() if isinstance(qm, QuadMesh)
    ]


@pytest.fixture
def seed() -> int:
    return 0


@pytest.fixture(scope="function")
def rng(seed: int) -> RandomState:
    """Create a RandomState to use in testing.

    This ensures the repeatability of histogram tests, that require large
    datasets that would be tedious to create manually.
    """
    return RandomState(seed)


@pytest.fixture
def normal_dist_2d(rng: RandomState) -> np.ndarray:
    """Points distributed by the standard multivariate normal.

    The standard multivariate normal is just two independent N(0, 1)
    distributions, one in each dimension.
    """
    samples = rng.multivariate_normal(
        (0.0, 0.0), [[1.0, 0.0], [0.0, 1.0]], (250, 3, 4)
    )
    return np.moveaxis(
        samples, 3, 1
    )  # Move generated space coords to correct axis position


@pytest.fixture
def histogram_data(normal_dist_2d: np.ndarray) -> xr.DataArray:
    """DataArray whose data is the ``normal_dist_2d`` points.

    Axes 2 and 3 are the individuals and keypoints axes, respectively.
    These dimensions are given coordinates {i,k}{0,1,2,3,4,5,...} for
    the purposes of indexing.
    """
    return xr.DataArray(
        data=normal_dist_2d,
        dims=["time", "space", "individuals", "keypoints"],
        coords={
            "space": ["x", "y"],
            "individuals": [f"i{i}" for i in range(normal_dist_2d.shape[2])],
            "keypoints": [f"k{i}" for i in range(normal_dist_2d.shape[3])],
        },
    )


@pytest.fixture
def histogram_data_with_nans(
    histogram_data: xr.DataArray, rng: RandomState
) -> xr.DataArray:
    """DataArray whose data is the ``normal_dist_2d`` points.

    Each datapoint has a chance of being turned into a NaN value.

    Axes 2 and 3 are the individuals and keypoints axes, respectively.
    These dimensions are given coordinates {i,k}{0,1,2,3,4,5,...} for
    the purposes of indexing.
    """
    data_with_nans = histogram_data.copy(deep=True)
    data_shape = data_with_nans.shape
    nan_chance = 1.0 / 25.0
    index_ranges = [range(dim_length) for dim_length in data_shape]
    for multiindex in product(*index_ranges):
        if rng.uniform() < nan_chance:
            data_with_nans[*multiindex] = float("nan")
    return data_with_nans


# def test_histogram_ignores_missing_dims(
#     input_does_not_have_dimensions: list[str],
# ) -> None:
#     """Test that ``occupancy_histogram`` ignores non-present dimensions."""
#     input_data = 0


@pytest.mark.parametrize(
    ["data", "individual", "keypoint", "n_bins"],
    [pytest.param("histogram_data", "i0", "k0", 30, id="30 bins each axis")],
)
def test_occupancy_histogram(
    data: xr.DataArray,
    individual: int | str,
    keypoint: int | str,
    n_bins: int | tuple[int, int],
    request,
) -> None:
    """Test that occupancy histograms correctly plot data."""
    if isinstance(data, str):
        data = request.getfixturevalue(data)

    plotted_hist = occupancy_histogram(
        data, individual=individual, keypoint=keypoint, bins=n_bins
    )

    # Confirm that a histogram was made
    plotted_data = get_histogram_binning_data(plotted_hist)
    assert len(plotted_data) == 1
    plotted_data = plotted_data[0]
    plotting_coords = plotted_data.get_coordinates()
    plotted_values = plotted_data.get_array()

    # Confirm the binned array has the correct size
    if not isinstance(n_bins, tuple):
        n_bins = (n_bins, n_bins)
    assert plotted_data.get_array().shape == n_bins

    # Confirm that each bin has the correct number of assignments
    data_time_xy = data.sel(individuals=individual, keypoints=keypoint)
    x_values = data_time_xy.sel(space="x").values
    y_values = data_time_xy.sel(space="y").values
    reconstructed_bins_limits_x = np.linspace(
        x_values.min(),
        x_values.max(),
        num=n_bins[0] + 1,
        endpoint=True,
    )
    assert all(
        np.allclose(reconstructed_bins_limits_x, plotting_coords[i, :, 0])
        for i in range(n_bins[0])
    )
    reconstructed_bins_limits_y = np.linspace(
        y_values.min(),
        y_values.max(),
        num=n_bins[1] + 1,
        endpoint=True,
    )
    assert all(
        np.allclose(reconstructed_bins_limits_y, plotting_coords[:, j, 1])
        for j in range(n_bins[1])
    )

    reconstructed_bin_counts = np.zeros(shape=n_bins, dtype=float)
    for i, xi in enumerate(reconstructed_bins_limits_x[:-1]):
        xi_p1 = reconstructed_bins_limits_x[i + 1]

        x_pts_in_range = (x_values >= xi) & (x_values <= xi_p1)
        for j, yj in enumerate(reconstructed_bins_limits_y[:-1]):
            yj_p1 = reconstructed_bins_limits_y[j + 1]

            y_pts_in_range = (y_values >= yj) & (y_values <= yj_p1)

            pts_in_this_bin = (x_pts_in_range & y_pts_in_range).sum()
            reconstructed_bin_counts[i, j] = pts_in_this_bin

            if pts_in_this_bin != plotted_values[i, j]:
                pass

    assert reconstructed_bin_counts.sum() == plotted_values.sum()
    assert np.all(reconstructed_bin_counts == plotted_values)
