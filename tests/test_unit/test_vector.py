import re
from collections.abc import Iterable
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


class TestComputeSignedAngle:
    """Tests for the compute_signed_angle_2d method."""

    x_axis = np.array([1.0, 0.0])
    y_axis = np.array([0.0, 1.0])
    coord_axes_array = np.array([x_axis, y_axis, -x_axis, -y_axis])

    @pytest.mark.parametrize(
        ["left_vector", "right_vector", "expected_angles"],
        [
            pytest.param(
                x_axis.reshape(1, 2),
                x_axis,
                [0.0],
                id="x-axis to x-axis",
            ),
            pytest.param(
                coord_axes_array,
                x_axis,
                [0.0, -np.pi / 2.0, np.pi, np.pi / 2.0],
                id="+/- axes to x-axis",
            ),
            pytest.param(
                coord_axes_array,
                -x_axis,
                [np.pi, np.pi / 2.0, 0.0, -np.pi / 2.0],
                id="+/- axes to -ve x-axis",
            ),
            pytest.param(
                np.array([[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]]),
                coord_axes_array,
                [-np.pi / 4.0] * 4,
                id="-pi/4 trailing axes",
            ),
            pytest.param(
                xr.DataArray(
                    data=coord_axes_array,
                    dims=["time", "space"],
                    coords={
                        "time": np.arange(coord_axes_array.shape[0]),
                        "space": ["x", "y"],
                    },
                ),
                xr.DataArray(
                    data=-1.0 * coord_axes_array,
                    dims=["time", "space"],
                    coords={
                        "time": np.arange(coord_axes_array.shape[0]),
                        "space": ["x", "y"],
                    },
                ),
                [np.pi] * 4,
                id="Two DataArrays given",
            ),
            pytest.param(
                np.array([-x_axis, x_axis]),
                np.array([x_axis, -x_axis]),
                [np.pi, np.pi],
                id="Rotation by '-pi' should map to pi.",
            ),
            pytest.param(
                np.array([-x_axis, x_axis]),
                np.zeros(shape=(3, 3, 3)),
                ValueError("v must be 1D or 2D, but got 3D"),
                id="Error: v is not 1 or 2 dimensional",
            ),
            pytest.param(
                np.array([-x_axis, x_axis]),
                "this is not an array",
                TypeError(
                    "v must be an xarray.DataArray or np.ndarray, "
                    "but got <class 'str'>"
                ),
                id="Error: v is not an array or DataArray",
            ),
            pytest.param(
                np.array([-x_axis, x_axis]),
                np.zeros((3, 2)),
                ValueError("conflicting sizes for dimension 'time'"),
                id="Error: v has incompatible shape",
            ),
        ],
    )
    def test_compute_signed_angle_2d(
        self,
        left_vector: xr.DataArray | np.ndarray,
        right_vector: xr.DataArray | np.ndarray,
        expected_angles: xr.DataArray | Exception,
    ) -> None:
        """Test computed angles are what we expect.

        This test also checks the antisymmetry of the function in question.
        Swapping the ``u`` and ``v`` arguments should produce an array of
        angles with the same magnitude but opposite signs
        (except for pi -> -pi).
        """
        if not isinstance(left_vector, xr.DataArray):
            left_vector = xr.DataArray(
                data=left_vector,
                dims=["time", "space"],
                coords={
                    "time": np.arange(left_vector.shape[0]),
                    "space": ["x", "y"],
                },
            )
        if isinstance(expected_angles, Exception):
            with pytest.raises(
                type(expected_angles), match=re.escape(str(expected_angles))
            ):
                vector.compute_signed_angle_2d(left_vector, right_vector)
        else:
            if not isinstance(expected_angles, xr.DataArray):
                expected_angles = xr.DataArray(
                    data=np.array(expected_angles),
                    dims=["time"],
                    coords={"time": left_vector["time"]}
                    if "time" in left_vector.dims
                    else None,
                )
            # pi and -pi should map to the same angle, regardless!
            expected_angles_reversed = expected_angles.copy(deep=True)
            expected_angles_reversed[expected_angles < np.pi] *= -1.0

            computed_angles = vector.compute_signed_angle_2d(
                left_vector, right_vector
            )
            computed_angles_reversed = vector.compute_signed_angle_2d(
                left_vector, right_vector, v_as_left_operand=True
            )

            xr.testing.assert_allclose(computed_angles, expected_angles)
            xr.testing.assert_allclose(
                computed_angles_reversed, expected_angles_reversed
            )

    @pytest.mark.parametrize(
        ["extra_dim_sizes"],
        [
            pytest.param((1,), id="[1D] Trailing singleton dimension"),
            pytest.param((2, 3), id="[2D] Individuals-keypoints esque"),
            pytest.param(
                (1, 2),
                id="[2D] As if .sel had been used on an additional axis",
            ),
        ],
    )
    def test_multidimensional_input(
        self, extra_dim_sizes: tuple[int, ...]
    ) -> None:
        """Test that non-spacetime dimensions are respected.

        The user may pass in data that has more than just the time and space
        dimensions, in which case the method should be able to cope with
        such an input, and preserve the additional dimensions.

        We simulate this case in this test by adding "junk" dimensions onto
        an existing input. We should see the result also spit out these "junk"
        dimensions, preserving the dimension names and coordinates too.
        """
        v_left = self.coord_axes_array.copy()
        v_left_original_ndim = v_left.ndim
        v_right = self.x_axis
        expected_angles = np.array([0.0, -np.pi / 2.0, np.pi, np.pi / 2.0])
        expected_angles_original_ndim = expected_angles.ndim

        # Append additional "junk" dimensions
        left_vector_coords: dict[str, Iterable[str] | Iterable[int]] = {
            "time": [f"t{i}" for i in range(v_left.shape[0])],
            "space": ["x", "y"],
        }
        for extra_dim_index, extra_dim_size in enumerate(extra_dim_sizes):
            v_left = np.repeat(
                v_left[..., np.newaxis],
                extra_dim_size,
                axis=extra_dim_index + v_left_original_ndim,
            )
            left_vector_coords[f"edim{extra_dim_index}"] = range(
                extra_dim_size
            )
            expected_angles = np.repeat(
                expected_angles[..., np.newaxis],
                extra_dim_size,
                axis=extra_dim_index + expected_angles_original_ndim,
            )
        v_left = xr.DataArray(
            data=v_left,
            dims=left_vector_coords.keys(),
            coords=left_vector_coords,
        )
        expected_angles_coords = dict(left_vector_coords)
        expected_angles_coords.pop("space")
        expected_angles = xr.DataArray(
            data=expected_angles,
            dims=expected_angles_coords.keys(),
            coords=expected_angles_coords,
        )

        computed_angles = vector.compute_signed_angle_2d(v_left, v_right)
        xr.testing.assert_allclose(computed_angles, expected_angles)
