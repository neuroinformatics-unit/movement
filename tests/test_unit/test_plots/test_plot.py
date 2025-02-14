import numpy as np
import pytest
import xarray as xr
from numpy.random import Generator, default_rng


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
def histogram_data_with_nans(histogram_data: xr.DataArray) -> xr.DataArray:
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
