from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
import xarray as xr

from movement.utils import vector


def compute_position_array_in_polar(position_array_cart):
    """Return an xarray.DataArray with position data in polar coordinates."""
    x_vals = position_array_cart.sel(space="x").values
    y_vals = position_array_cart.sel(space="y").values

    # Compute polar coordinates
    rho = np.sqrt(x_vals**2 + y_vals**2)
    phi = np.arctan2(y_vals, x_vals)

    # Build position array with space in polar coordinates
    return xr.DataArray(
        np.column_stack((rho, phi)),
        dims=["time", "space_pol"],
        coords={
            "time": position_array_cart.coords["time"],
            "space_pol": ["rho", "phi"],
        },
    )


# ---- trajectories in Cartesian coordinates ----
@pytest.fixture
def trajectory_x_eq_0_cart():
    """Return an xarray.DataArray with position data in Cartesian coordinates.

    A particle that moves along the y positive axis from the origin (x=0),
    1 pixel per frame.
    Uniform linear motion y axis
    """
    # Define trajectory
    n_frames = 10
    x_vals = np.zeros((n_frames,))
    y_vals = np.arange(n_frames)

    # Build position array with space in cartesian coordinates
    return xr.DataArray(
        np.column_stack((x_vals, y_vals)),
        dims=["time", "space"],
        coords={"time": np.arange(n_frames), "space": ["x", "y"]},
    )


@pytest.fixture
def trajectory_x_eq_y_cart():
    """Return an xarray.DataArray with position data in Cartesian coordinates.

    A particle that moves along the y positive axis from the origin (x=0),
    1 pixel per frame.
    Uniform linear motion y axis
    """
    # Define trajectory
    n_frames = 10
    x_vals = np.arange(n_frames)
    y_vals = x_vals.copy()

    # Build position array with space in cartesian coordinates
    return xr.DataArray(
        np.column_stack((x_vals, y_vals)),
        dims=["time", "space"],
        coords={"time": np.arange(n_frames), "space": ["x", "y"]},
    )


@pytest.fixture
def trajectory_random_cart():
    """Return an xarray.DataArray with position data in Cartesian coordinates.

    A particle that moves randomly in the xy plane.
    """
    # Cartesian coordinates
    x_vals = np.array([0.0, -1.0, 0.0, 1.0, 1.0, 1.0, 0.0, -1.0, -10.0])
    y_vals = np.array([0.0, -1.0, -1.0, -1.0, 0.0, 1.0, 1.0, 1.0, 0.0])
    time_coords = np.arange(len(x_vals))

    # Build position array with space in cartesian coordinates
    return xr.DataArray(
        np.column_stack((x_vals, y_vals)),
        dims=["time", "space"],
        coords={"time": time_coords, "space": ["x", "y"]},
    )


# ---- trajectories in polar coordinates ----
@pytest.fixture  # use param here too?
def trajectory_x_eq_0_pol(trajectory_x_eq_0_cart):
    return compute_position_array_in_polar(trajectory_x_eq_0_cart)


@pytest.fixture
def trajectory_x_eq_y_pol(trajectory_x_eq_y_cart):
    return compute_position_array_in_polar(trajectory_x_eq_y_cart)


@pytest.fixture
def trajectory_random_pol(trajectory_random_cart):
    return compute_position_array_in_polar(trajectory_random_cart)


# ---- with nan values ----
@pytest.fixture(
    params=[
        "trajectory_x_eq_0_cart",
        "trajectory_x_eq_y_cart",
        "trajectory_random_cart",
    ]
)
def trajectories_cart_with_nan(request):
    trajectory_data_array = request.getfixturevalue(request.param)
    trajectory_data_array.loc[{"time": slice(2, 3)}] = np.nan
    return trajectory_data_array


@pytest.fixture(
    params=[
        "trajectory_x_eq_0_pol",
        "trajectory_x_eq_y_pol",
        "trajectory_random_pol",
    ]
)
def trajectories_pol_with_nan(request):
    trajectory_data_array = request.getfixturevalue(request.param)
    trajectory_data_array.loc[{"time": slice(2, 3)}] = np.nan
    return trajectory_data_array


# ---- invalid data arrays ----
@pytest.fixture
def trajectory_cart_with_missing_space_dim(trajectory_random_cart):
    """Return an xarray.Dataset with Cartesian and polar coordinates,
    where the required ``space`` dimension is missing.
    """
    return trajectory_random_cart.rename({"space": "spice"})


@pytest.fixture
def trajectory_cart_with_missing_space_coord(trajectory_random_cart):
    """Return an xarray.DataArray where the required ``space["x"]`` and
    ``space["y"]`` coordinates are missing.
    """
    trajectory_random_cart["space"] = ["a", "b"]
    return trajectory_random_cart


@pytest.mark.parametrize(
    "position_data_array_cart, expected_rho, expected_phi",
    [
        ("trajectory_x_eq_0_cart", np.arange(10), [0.0] + [np.pi / 2] * 9),
        (
            "trajectory_x_eq_y_cart",
            np.sqrt(2) * np.arange(10),
            [0.0] + [np.arctan(1)] * 9,
        ),
        (
            "trajectory_random_cart",
            np.array(
                [
                    0.0,
                    np.sqrt(2),
                    1,
                    np.sqrt(2),
                    1,
                    np.sqrt(2),
                    1,
                    np.sqrt(2),
                    10,
                ]
            ),
            np.pi
            * np.array([0.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]),
        ),
    ],
)
def test_cart2pol(
    position_data_array_cart, expected_rho, expected_phi, request
):
    """Test Cartesian to polar coordinates with known values."""
    position_array = request.getfixturevalue(position_data_array_cart)
    position_array_pol = vector.cart2pol(position_array)

    assert np.allclose(
        position_array_pol.sel(space_pol="rho").values,
        expected_rho,
    )

    assert np.allclose(
        position_array_pol.sel(space_pol="phi").values,
        expected_phi,
    )


def test_cart2pol_with_nan(trajectories_cart_with_nan):
    """Test Cartesian to polar coordinates with NaN values."""
    position_array_pol = vector.cart2pol(trajectories_cart_with_nan)

    n_expected_nans_per_coord = 2

    # Check that NaN values are preserved
    assert (
        sum(np.isnan(position_array_pol.sel(space_pol="rho")))
        == n_expected_nans_per_coord
    )
    assert (
        sum(np.isnan(position_array_pol.sel(space_pol="phi")))
        == n_expected_nans_per_coord
    )

    @pytest.mark.parametrize(
        "ds, expected_exception",
        [
            ("cart_pol_dataset", does_not_raise()),
            ("cart_pol_dataset_with_nan", does_not_raise()),
            (
                "cart_pol_dataset_missing_pol_dim",
                pytest.raises(ValueError),
            ),
            (
                "cart_pol_dataset_missing_pol_coords",
                pytest.raises(ValueError),
            ),
        ],
    )
    def test_pol2cart(self, ds, expected_exception, request):
        """Test polar to Cartesian coordinates with known values."""
        ds = request.getfixturevalue(ds)
        with expected_exception:
            result = vector.pol2cart(ds.pol)
            xr.testing.assert_allclose(result, ds.cart)

    @pytest.mark.parametrize(
        "ds, expected_exception",
        [
            ("cart_pol_dataset", does_not_raise()),
            ("cart_pol_dataset_with_nan", does_not_raise()),
            ("cart_pol_dataset_missing_cart_dim", pytest.raises(ValueError)),
            (
                "cart_pol_dataset_missing_cart_coords",
                pytest.raises(ValueError),
            ),
        ],
    )
    def test_compute_norm(self, ds, expected_exception, request):
        """Test vector norm computation with known values."""
        ds = request.getfixturevalue(ds)
        with expected_exception:
            # validate the norm computation
            result = vector.compute_norm(ds.cart)
            expected = np.sqrt(
                ds.cart.sel(space="x") ** 2 + ds.cart.sel(space="y") ** 2
            )
            xr.testing.assert_allclose(result, expected)

            # result should be the same from Cartesian and polar coordinates
            xr.testing.assert_allclose(result, vector.compute_norm(ds.pol))

            # The result should only contain the time dimension.
            assert result.dims == ("time",)

    @pytest.mark.parametrize(
        "ds, expected_exception",
        [
            ("cart_pol_dataset", does_not_raise()),
            ("cart_pol_dataset_with_nan", does_not_raise()),
            ("cart_pol_dataset_missing_cart_dim", pytest.raises(ValueError)),
        ],
    )
    def test_convert_to_unit(self, ds, expected_exception, request):
        """Test conversion to unit vectors (normalisation)."""
        ds = request.getfixturevalue(ds)
        with expected_exception:
            # normalise both the Cartesian and the polar data to unit vectors
            unit_cart = vector.convert_to_unit(ds.cart)
            unit_pol = vector.convert_to_unit(ds.pol)
            # they should yield the same result, just in different coordinates
            xr.testing.assert_allclose(unit_cart, vector.pol2cart(unit_pol))
            xr.testing.assert_allclose(unit_pol, vector.cart2pol(unit_cart))

            # since we established that polar vs Cartesian unit vectors are
            # equivalent, it's enough to do other assertions on either one

            # the normalised data should have the same dimensions as the input
            assert unit_cart.dims == ds.cart.dims

            # unit vector should be NaN if the input vector was null or NaN
            is_null_vec = (ds.cart == 0).all("space")  # null vec: x=0, y=0
            is_nan_vec = ds.cart.isnull().any("space")  # any NaN in x or y
            expected_nan_idxs = is_null_vec | is_nan_vec
            assert unit_cart.where(expected_nan_idxs).isnull().all()

            # For non-NaN unit vectors in polar coordinates, the rho values
            # should be 1 and the phi values should be the same as the input
            expected_unit_pol = ds.pol.copy()
            expected_unit_pol.loc[{"space_pol": "rho"}] = 1
            expected_unit_pol = expected_unit_pol.where(~expected_nan_idxs)
            xr.testing.assert_allclose(unit_pol, expected_unit_pol)
