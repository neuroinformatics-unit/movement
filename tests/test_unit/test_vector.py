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
