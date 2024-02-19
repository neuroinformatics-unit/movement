import numpy as np
import pytest
import xarray as xr

from movement.analysis import kinematics


class TestKinematics:
    """Test suite for the kinematics module."""

    @pytest.fixture
    def expected_dataarray(self, valid_pose_dataset):
        """Return a function to generate the expected dataarray
        for different kinematic properties."""

        def _expected_dataarray(property):
            """Return an xarray.DataArray with default values and
            the expected dimensions and coordinates."""
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
            # Repeat the x_vals to match the shape of the pose_tracks
            x_vals = np.tile(x_vals, (1, 2, 2, 1))
            return xr.DataArray(
                np.concatenate(
                    [x_vals, y_vals],
                    axis=-1,
                ),
                dims=valid_pose_dataset.dims,
                coords=valid_pose_dataset.coords,
            )

        return _expected_dataarray

    def test_displacement(self, valid_pose_dataset, expected_dataarray):
        """Test displacement computation."""
        result = kinematics.compute_displacement(
            valid_pose_dataset.pose_tracks
        )
        xr.testing.assert_allclose(result, expected_dataarray("displacement"))

    def test_velocity(self, valid_pose_dataset, expected_dataarray):
        """Test velocity computation."""
        result = kinematics.compute_velocity(valid_pose_dataset.pose_tracks)
        xr.testing.assert_allclose(result, expected_dataarray("velocity"))

    def test_acceleration(self, valid_pose_dataset, expected_dataarray):
        """Test acceleration computation."""
        result = kinematics.compute_acceleration(
            valid_pose_dataset.pose_tracks
        )
        xr.testing.assert_allclose(result, expected_dataarray("acceleration"))

    @pytest.mark.parametrize("order", [0, -1, 1.0, "1"])
    def test_approximate_derivative_with_invalid_order(self, order):
        """Test that an error is raised when the order is non-positive."""
        data = np.arange(10)
        expected_exception = (
            ValueError if isinstance(order, int) else TypeError
        )
        with pytest.raises(expected_exception):
            kinematics.compute_approximate_derivative(data, order=order)

    def test_compute_with_missing_time_dimension(
        self, missing_dim_dataset, kinematic_property
    ):
        """Test that computing a property of a pose dataset with
        missing 'time' dimension raises the appropriate error."""
        with pytest.raises(ValueError):
            eval(f"kinematics.compute_{kinematic_property}")(
                missing_dim_dataset
            )
