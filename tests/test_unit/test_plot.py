import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr
from matplotlib.collections import QuadMesh
from numpy.random import Generator, default_rng

from movement.plot import occupancy_histogram


@pytest.fixture
def seed() -> int:
    return 0


@pytest.fixture(scope="function")
def rng(seed: int) -> Generator:
    """Create a RandomState to use in testing.

    This ensures the repeatability of histogram tests, that require large
    datasets that would be tedious to create manually.
    """
    return default_rng(seed)


@pytest.fixture
def normal_dist_2d(rng: Generator) -> np.ndarray:
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
    histogram_data: xr.DataArray, rng: Generator
) -> xr.DataArray:
    """DataArray whose data is the ``normal_dist_2d`` points.

    Axes 2 and 3 are the individuals and keypoints axes, respectively.
    These dimensions are given coordinates {i,k}{0,1,2,3,4,5,...} for
    the purposes of indexing.

    For individual i0, keypoint k0, the following (time, space) values are
    converted into NaNs:
    - (100, "x")
    - (200, "y")
    - (150, "x")
    - (150, "y")

    """
    individual_0 = "i0"
    keypoint_0 = "k0"
    data_with_nans = histogram_data.copy(deep=True)
    for time_index, space_coord in [
        (100, "x"),
        (200, "y"),
        (150, "x"),
        (150, "y"),
    ]:
        data_with_nans.loc[
            time_index, space_coord, individual_0, keypoint_0
        ] = float("nan")
    return data_with_nans


@pytest.fixture
def entirely_nan_data(histogram_data: xr.DataArray) -> xr.DataArray:
    return histogram_data.copy(
        deep=True, data=histogram_data.values * float("nan")
    )


@pytest.mark.parametrize(
    [
        "data",
        "remove_dims_from_data_before_starting",
        "individual",
        "keypoint",
        "n_bins",
    ],
    [
        pytest.param(
            "histogram_data",
            [],
            "i0",
            "k0",
            30,
            id="30 bins each axis",
        ),
        pytest.param(
            "histogram_data",
            [],
            "i1",
            "k0",
            (20, 30),
            id="(20, 30) bins",
        ),
        pytest.param(
            "histogram_data_with_nans",
            [],
            "i0",
            "k0",
            30,
            id="NaNs should be removed",
        ),
        pytest.param(
            "entirely_nan_data",
            [],
            "i0",
            "k0",
            10,
            id="All NaN-data",
        ),
        pytest.param(
            "histogram_data",
            ["individuals"],
            "i0",
            "k0",
            30,
            id="Ignores individual if not a dimension",
        ),
        pytest.param(
            "histogram_data",
            ["keypoints"],
            "i0",
            "k1",
            30,
            id="Ignores keypoint if not a dimension",
        ),
        pytest.param(
            "histogram_data",
            ["individuals", "keypoints"],
            "i0",
            "k0",
            30,
            id="Can handle raw xy data",
        ),
    ],
)
def test_occupancy_histogram(
    data: xr.DataArray,
    remove_dims_from_data_before_starting: list[str],
    individual: int | str,
    keypoint: int | str,
    n_bins: int | tuple[int, int],
    request,
) -> None:
    """Test that occupancy histograms correctly plot data.

    Specifically, check that:
    - The bin edges are what we expect.
    - The bin counts can be manually verified and are in agreement.
    - Only non-NaN values are plotted, but NaN values do not throw errors.
    """
    if isinstance(data, str):
        data = request.getfixturevalue(data)

    # We will need to only select the xy data later in the test,
    # but if we are dropping dimensions we might need to call it
    # in different ways.
    kwargs_to_select_xy_data = {
        "individuals": individual,
        "keypoints": keypoint,
    }
    for d in remove_dims_from_data_before_starting:
        # Retain the 0th value in the corresponding dimension,
        # then drop that dimension.
        data = data.sel({d: getattr(data, d)[0]}).squeeze()
        assert d not in data.dims

        # We no longer need to filter this dimension out
        # when examining the xy data later in the test.
        kwargs_to_select_xy_data.pop(d, None)

    _, _, histogram_info = occupancy_histogram(
        data, individual=individual, keypoint=keypoint, bins=n_bins
    )
    plotted_values = histogram_info["counts"]

    # Confirm the binned array has the correct size
    if not isinstance(n_bins, tuple):
        n_bins = (n_bins, n_bins)
    assert plotted_values.shape == n_bins

    # Confirm that each bin has the correct number of assignments
    data_time_xy = data.sel(**kwargs_to_select_xy_data)
    data_time_xy = data_time_xy.dropna(dim="time", how="any")
    plotted_x_values = data_time_xy.sel(space="x").values
    plotted_y_values = data_time_xy.sel(space="y").values
    assert plotted_x_values.shape == plotted_y_values.shape
    # This many non-NaN values were plotted
    n_non_nan_values = plotted_x_values.shape[0]

    if n_non_nan_values > 0:
        reconstructed_bins_limits_x = np.linspace(
            plotted_x_values.min(),
            plotted_x_values.max(),
            num=n_bins[0] + 1,
            endpoint=True,
        )
        assert np.allclose(
            reconstructed_bins_limits_x, histogram_info["xedges"]
        )
        reconstructed_bins_limits_y = np.linspace(
            plotted_y_values.min(),
            plotted_y_values.max(),
            num=n_bins[1] + 1,
            endpoint=True,
        )
        assert np.allclose(
            reconstructed_bins_limits_y, histogram_info["yedges"]
        )

        reconstructed_bin_counts = np.zeros(shape=n_bins, dtype=float)
        for i, xi in enumerate(reconstructed_bins_limits_x[:-1]):
            xi_p1 = reconstructed_bins_limits_x[i + 1]

            x_pts_in_range = (plotted_x_values >= xi) & (
                plotted_x_values <= xi_p1
            )
            for j, yj in enumerate(reconstructed_bins_limits_y[:-1]):
                yj_p1 = reconstructed_bins_limits_y[j + 1]

                y_pts_in_range = (plotted_y_values >= yj) & (
                    plotted_y_values <= yj_p1
                )

                pts_in_this_bin = (x_pts_in_range & y_pts_in_range).sum()
                reconstructed_bin_counts[i, j] = pts_in_this_bin

        # We agree with a manual count
        assert reconstructed_bin_counts.sum() == plotted_values.sum()
        # All non-NaN values were plotted
        assert n_non_nan_values == plotted_values.sum()
        # The counts were actually correct
        assert np.all(reconstructed_bin_counts == plotted_values)
    else:
        # No non-nan values were given
        assert plotted_values.sum() == 0


def test_respects_axes(histogram_data: xr.DataArray) -> None:
    """Check that existing axes objects are respected if passed."""
    # Plotting on existing axes
    _, existing_ax = plt.subplots(1, 2)

    existing_ax[0].plot(
        np.linspace(0.0, 10.0, num=100), np.linspace(0.0, 10.0, num=100)
    )

    _, _, hist_info_existing = occupancy_histogram(
        histogram_data, ax=existing_ax[1]
    )
    hist_plots_added = [
        qm for qm in existing_ax[1].get_children() if isinstance(qm, QuadMesh)
    ]
    assert len(hist_plots_added) == 1

    # Plot on new axis and create a new figure
    new_fig, new_ax, hist_info_new = occupancy_histogram(histogram_data)
    assert isinstance(new_fig, plt.Figure)
    assert isinstance(new_ax, plt.Axes)
    hist_plots_created = [
        qm for qm in new_ax.get_children() if isinstance(qm, QuadMesh)
    ]
    assert len(hist_plots_created) == 1

    # Check that the same plot was made for each
    assert set(hist_info_new.keys()) == set(hist_info_existing.keys())
    for key, new_ax_value in hist_info_new.items():
        existing_ax_value = hist_info_existing[key]

        assert np.allclose(new_ax_value, existing_ax_value)
