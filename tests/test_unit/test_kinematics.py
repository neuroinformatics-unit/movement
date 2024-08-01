from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
import xarray as xr

from movement.analysis import kinematics


class TestKinematics:
    """Test suite for the kinematics module."""

    @pytest.fixture
    def expected_dataarray(self, valid_poses_dataset):
        """Return a function to generate the expected dataarray
        for different kinematic properties.
        """

        def _expected_dataarray(property):
            """Return an xarray.DataArray with default values and
            the expected dimensions and coordinates.
            """
            # Expected x,y values for velocity
            x_vals = np.array(
                [1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 17.0]
            )
            y_vals = np.full((10, 2, 2, 1), 4.0)
            if property == "acceleration":
                x_vals = np.array(
                    [1.0, 1.5, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.5, 1.0]
                )
                y_vals = np.full((10, 2, 2, 1), 0)
            elif property == "displacement":
                x_vals = np.array(
                    [0.0, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0]
                )
                y_vals[0] = 0

            x_vals = x_vals.reshape(-1, 1, 1, 1)
            # Repeat the x_vals to match the shape of the position
            x_vals = np.tile(x_vals, (1, 2, 2, 1))
            return xr.DataArray(
                np.concatenate(
                    [x_vals, y_vals],
                    axis=-1,
                ),
                dims=valid_poses_dataset.dims,
                coords=valid_poses_dataset.coords,
            )

        return _expected_dataarray

    kinematic_test_params = [
        ("valid_poses_dataset", does_not_raise()),
        ("valid_poses_dataset_with_nan", does_not_raise()),
        ("missing_dim_poses_dataset", pytest.raises(AttributeError)),
    ]

    @pytest.mark.parametrize("ds, expected_exception", kinematic_test_params)
    def test_displacement(
        self, ds, expected_exception, expected_dataarray, request
    ):
        """Test displacement computation."""
        ds = request.getfixturevalue(ds)
        with expected_exception:
            result = kinematics.compute_displacement(ds.position)
            expected = expected_dataarray("displacement")
            if ds.position.isnull().any():
                expected.loc[
                    {"individuals": "ind1", "time": [3, 4, 7, 8, 9]}
                ] = np.nan
            xr.testing.assert_allclose(result, expected)

    @pytest.mark.parametrize("ds, expected_exception", kinematic_test_params)
    def test_velocity(
        self, ds, expected_exception, expected_dataarray, request
    ):
        """Test velocity computation."""
        ds = request.getfixturevalue(ds)
        with expected_exception:
            result = kinematics.compute_velocity(ds.position)
            expected = expected_dataarray("velocity")
            if ds.position.isnull().any():
                expected.loc[
                    {"individuals": "ind1", "time": [2, 4, 6, 7, 8, 9]}
                ] = np.nan
            xr.testing.assert_allclose(result, expected)

    @pytest.mark.parametrize("ds, expected_exception", kinematic_test_params)
    def test_acceleration(
        self, ds, expected_exception, expected_dataarray, request
    ):
        """Test acceleration computation."""
        ds = request.getfixturevalue(ds)
        with expected_exception:
            result = kinematics.compute_acceleration(ds.position)
            expected = expected_dataarray("acceleration")
            if ds.position.isnull().any():
                expected.loc[
                    {"individuals": "ind1", "time": [1, 3, 5, 6, 7, 8, 9]}
                ] = np.nan
            xr.testing.assert_allclose(result, expected)

    @pytest.mark.parametrize("order", [0, -1, 1.0, "1"])
    def test_approximate_derivative_with_invalid_order(self, order):
        """Test that an error is raised when the order is non-positive."""
        data = np.arange(10)
        expected_exception = (
            ValueError if isinstance(order, int) else TypeError
        )
        with pytest.raises(expected_exception):
            kinematics._compute_approximate_derivative(data, order=order)
