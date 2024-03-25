from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
import xarray as xr

from movement.analysis import vector_utils


class TestVectorUtils:
    """Test suite for the vector_utils module."""

    @pytest.fixture
    def cart_pol_dataset(self):
        def _cart_pol_dataset(ds_type):
            """Return an xarray.Dataset with Cartesian and polar
            coordinates, and expected exception depending on the
            type of requested."""
            x_vals = np.array([1, 1, 0, -1, -10, -1, 0, 1], dtype=float)
            y_vals = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=float)
            time_coords = np.arange(len(x_vals))
            rho = np.sqrt(x_vals**2 + y_vals**2)
            theta = np.pi * np.array(
                [0, 0.25, 0.5, 0.75, 1, -0.75, -0.5, -0.25]
            )
            cart = xr.DataArray(
                np.column_stack((x_vals, y_vals)),
                dims=["time", "space"],
                coords={"time": time_coords, "space": ["x", "y"]},
            )
            pol = xr.DataArray(
                np.column_stack((rho, theta)),
                dims=["time", "space_polar"],
                coords={
                    "time": time_coords,
                    "space_polar": ["rho", "theta"],
                },
            )
            ds = xr.Dataset(
                data_vars={
                    "cart": cart,
                    "pol": pol,
                },
            )
            ds.attrs.update({"expected_exception": does_not_raise()})
            if ds_type == "contains_nan":
                ds.cart.loc[{"time": slice(2, 3)}] = np.nan
                ds.pol.loc[{"time": slice(2, 3)}] = np.nan
            elif ds_type == "missing_cart_dim":
                ds = ds.rename({"space": "spice"})
                ds.attrs.update(
                    {"expected_exception": pytest.raises(ValueError)}
                )
            elif ds_type == "missing_cart_coords":
                ds["space"] = ["a", "b"]
                ds.attrs.update(
                    {"expected_exception": pytest.raises(ValueError)}
                )
            elif ds_type == "missing_polar_dim":
                ds = ds.rename({"space_polar": "spice_polar"})
                ds.attrs.update(
                    {"expected_exception": pytest.raises(ValueError)}
                )
            elif ds_type == "missing_polar_coords":
                ds["space_polar"] = ["a", "b"]
                ds.attrs.update(
                    {"expected_exception": pytest.raises(ValueError)}
                )
            return ds

        return _cart_pol_dataset

    @pytest.mark.parametrize(
        "ds_type",
        [None, "contains_nan", "missing_cart_dim", "missing_cart_coords"],
    )
    def test_cart2pol(self, cart_pol_dataset, ds_type):
        """Test Cartesian to polar coordinates with known values."""
        ds = cart_pol_dataset(ds_type)
        with ds.expected_exception:
            result = vector_utils.cart2pol(ds.cart)
            xr.testing.assert_allclose(result, ds.pol)

    @pytest.mark.parametrize(
        "ds_type",
        [None, "contains_nan", "missing_polar_dim", "missing_polar_coords"],
    )
    def test_pol2cart(self, cart_pol_dataset, ds_type):
        """Test polar to Cartesian coordinates with known values."""
        ds = cart_pol_dataset(ds_type)
        with ds.expected_exception:
            result = vector_utils.pol2cart(ds.pol)
            xr.testing.assert_allclose(result, ds.cart)
