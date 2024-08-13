from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
import xarray as xr

from movement.utils import vector


class TestVector:
    """Test suite for the utils.vector module."""

    @pytest.fixture
    def cart_pol_dataset(self):
        """Return an xarray.Dataset with Cartesian and polar coordinates."""
        x_vals = np.array([-0.0, -1.0, 0.0, 1.0, 1.0, 1.0, 0.0, -1.0, -10.0])
        y_vals = np.array([-0.0, -1.0, -1.0, -1.0, 0.0, 1.0, 1.0, 1.0, 0.0])
        time_coords = np.arange(len(x_vals))
        rho = np.sqrt(x_vals**2 + y_vals**2)
        phi = np.pi * np.array(
            [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
        )
        cart = xr.DataArray(
            np.column_stack((x_vals, y_vals)),
            dims=["time", "space"],
            coords={"time": time_coords, "space": ["x", "y"]},
        )
        pol = xr.DataArray(
            np.column_stack((rho, phi)),
            dims=["time", "space_pol"],
            coords={
                "time": time_coords,
                "space_pol": ["rho", "phi"],
            },
        )
        return xr.Dataset(
            data_vars={
                "cart": cart,
                "pol": pol,
            },
        )

    @pytest.fixture
    def cart_pol_dataset_with_nan(self, cart_pol_dataset):
        """Return an xarray.Dataset with Cartesian and polar coordinates,
        where some values are NaN.
        """
        cart_pol_dataset.cart.loc[{"time": slice(2, 3)}] = np.nan
        cart_pol_dataset.pol.loc[{"time": slice(2, 3)}] = np.nan
        return cart_pol_dataset

    @pytest.fixture
    def cart_pol_dataset_missing_cart_dim(self, cart_pol_dataset):
        """Return an xarray.Dataset with Cartesian and polar coordinates,
        where the required ``space`` dimension is missing.
        """
        return cart_pol_dataset.rename({"space": "spice"})

    @pytest.fixture
    def cart_pol_dataset_missing_cart_coords(self, cart_pol_dataset):
        """Return an xarray.Dataset with Cartesian and polar coordinates,
        where the required ``space["x"]`` and ``space["y"]`` coordinates
        are missing.
        """
        cart_pol_dataset["space"] = ["a", "b"]
        return cart_pol_dataset

    @pytest.fixture
    def cart_pol_dataset_missing_pol_dim(self, cart_pol_dataset):
        """Return an xarray.Dataset with Cartesian and polar coordinates,
        where the required ``space_pol`` dimension is missing.
        """
        return cart_pol_dataset.rename({"space_pol": "spice_pol"})

    @pytest.fixture
    def cart_pol_dataset_missing_pol_coords(self, cart_pol_dataset):
        """Return an xarray.Dataset with Cartesian and polar coordinates,
        where the required ``space_pol["rho"]`` and ``space_pol["phi"]``
        coordinates are missing.
        """
        cart_pol_dataset["space_pol"] = ["a", "b"]
        return cart_pol_dataset

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
    def test_cart2pol(self, ds, expected_exception, request):
        """Test Cartesian to polar coordinates with known values."""
        ds = request.getfixturevalue(ds)
        with expected_exception:
            result = vector.cart2pol(ds.cart)
            xr.testing.assert_allclose(result, ds.pol)

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
            unit_cart = vector.convert_to_unit(ds.cart)  # normalise Cartesian
            unit_pol = vector.convert_to_unit(ds.pol)  # normalise polar

            # the normalised data should have the same dimensions as the input
            assert unit_cart.dims == ds.cart.dims
            assert unit_pol.dims == ds.pol.dims

            # normalising null vectors (where x,y = 0,0) should yield NaNs
            is_null_vector = (ds.cart == 0).all("space")
            assert unit_cart.where(is_null_vector).isnull().all()
            assert unit_pol.where(is_null_vector).isnull().all()

            # the norm of the unit vectors should be 1 for all
            # time points except for the expected NaNs.
            unit_cart_norm = vector.compute_norm(unit_cart).values
            expected_norm = np.ones_like(unit_cart_norm)
            expected_norm[unit_cart.isnull().any("space")] = np.nan
            np.testing.assert_allclose(unit_cart_norm, expected_norm)

            # normalising the polar data should yield the same result
            # as converting the normalised Cartesian data to polar.
            xr.testing.assert_allclose(unit_pol, vector.cart2pol(unit_cart))

            # normalising the Cartesian data should yield the same result
            # as converting the normalised polar data to Cartesian.
            xr.testing.assert_allclose(unit_cart, vector.pol2cart(unit_pol))
