import numpy as np
import pytest
import xarray as xr

from movement.analysis import kinematics


class TestKinematics:
    """Test suite for the kinematics module."""

    def test_displacement(self, valid_pose_dataset):
        """Test displacement calculation."""
        data = valid_pose_dataset.pose_tracks
        result = kinematics.displacement(data)
        expected_values = np.full((10, 2, 2, 2), [3.0, 4.0])
        expected_values[0, :, :, :] = 0
        expected = xr.DataArray(
            expected_values,
            dims=data.dims,
            coords=data.coords,
        )
        xr.testing.assert_allclose(result, expected)

    def test_displacement_vector(self, valid_pose_dataset):
        """Test displacement vector calculation."""
        data = valid_pose_dataset.pose_tracks
        result = kinematics.displacement_vector(data)
        expected_magnitude = np.full((10, 2, 2), 5.0)
        expected_magnitude[0, :, :] = 0
        expected_direction = np.full((10, 2, 2), 0.92729522)
        expected_direction[0, :, :] = 0
        expected = xr.Dataset(
            data_vars={
                "magnitude": xr.DataArray(
                    expected_magnitude, dims=data.dims[:-1]
                ),
                "direction": xr.DataArray(
                    expected_direction, dims=data.dims[:-1]
                ),
            },
            coords={
                "time": data.time,
                "keypoints": data.keypoints,
                "individuals": data.individuals,
            },
        )
        xr.testing.assert_allclose(result, expected)

    def test_velocity(self, valid_pose_dataset):
        """Test velocity calculation."""
        data = valid_pose_dataset.pose_tracks
        # Compute velocity
        result = kinematics.velocity(data)
        expected_values = np.full((10, 2, 2, 2), [3.0, 4.0])
        expected = xr.DataArray(
            expected_values,
            dims=data.dims,
            coords=data.coords,
        )
        xr.testing.assert_allclose(result, expected)

    def test_velocity_vector(self, valid_pose_dataset):
        """Test velocity vector calculation."""
        data = valid_pose_dataset.pose_tracks
        result = kinematics.velocity_vector(data)
        expected_magnitude = np.full((10, 2, 2), 5.0)
        expected_direction = np.full((10, 2, 2), 0.92729522)
        expected = xr.Dataset(
            data_vars={
                "magnitude": xr.DataArray(
                    expected_magnitude, dims=data.dims[:-1]
                ),
                "direction": xr.DataArray(
                    expected_direction, dims=data.dims[:-1]
                ),
            },
            coords={
                "time": data.time,
                "keypoints": data.keypoints,
                "individuals": data.individuals,
            },
        )
        xr.testing.assert_allclose(result, expected)

    def test_acceleration(self, valid_pose_dataset):
        """Test acceleration calculation."""
        data = valid_pose_dataset.pose_tracks
        result = kinematics.acceleration(data)
        expected = xr.DataArray(
            np.zeros((10, 2, 2, 2)),
            dims=data.dims,
            coords=data.coords,
        )
        xr.testing.assert_allclose(result, expected)

    def test_acceleration_vector(self, valid_pose_dataset):
        """Test acceleration vector calculation."""
        data = valid_pose_dataset.pose_tracks
        result = kinematics.acceleration_vector(data)
        expected = xr.Dataset(
            data_vars={
                "magnitude": xr.DataArray(
                    np.zeros((10, 2, 2)), dims=data.dims[:-1]
                ),
                "direction": xr.DataArray(
                    np.zeros((10, 2, 2)), dims=data.dims[:-1]
                ),
            },
            coords={
                "time": data.time,
                "keypoints": data.keypoints,
                "individuals": data.individuals,
            },
        )
        xr.testing.assert_allclose(result, expected)

    def test_approximate_derivative_with_nonpositive_order(self):
        """Test that an error is raised when the order is non-positive."""
        data = np.arange(10)
        with pytest.raises(ValueError):
            kinematics.approximate_derivative(data, order=0)
