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
    y_vals = np.arange(n_frames, dtype=float)

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
    x_vals = np.arange(n_frames, dtype=float)
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
    "invalid_position_array",
    [
        "trajectory_cart_with_missing_space_dim",
        "trajectory_cart_with_missing_space_coord",
    ],
)
def test_cart2pol_invalid_array(invalid_position_array, request):
    """Test Cartesian to polar coordinates with invalid input."""
    with pytest.raises(ValueError) as excinfo:
        vector.cart2pol(request.getfixturevalue(invalid_position_array))

    assert (
        "Input data must contain ['x', 'y'] in the 'space' coordinates."
        in str(excinfo.value)
    )
