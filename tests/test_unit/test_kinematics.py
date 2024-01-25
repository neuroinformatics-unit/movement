import numpy as np
import pytest
import xarray as xr

from movement.analysis import kinematics


class TestKinematics:
    """Test suite for the kinematics module."""

    def test_distance(self, valid_pose_dataset):
        """Test distance calculation."""
        data = valid_pose_dataset.pose_tracks
        result = kinematics.distance(data)
        expected = np.full((10, 2, 2), 5.0)
        expected[0, :, :] = np.nan
        np.testing.assert_allclose(result.values, expected)

    def test_displacement(self, valid_pose_dataset):
        """Test displacement calculation."""
        data = valid_pose_dataset.pose_tracks
        result = kinematics.displacement(data)
        expected_magnitude = np.full((10, 2, 2), 5.0)
        expected_magnitude[0, :, :] = np.nan
        expected_direction = np.full((10, 2, 2), 0.92729522)
        expected_direction[0, :, :] = np.nan
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

    def test_speed(self, valid_pose_dataset):
        """Test speed calculation."""
        data = valid_pose_dataset.pose_tracks
        result = kinematics.speed(data)
        expected = np.full((10, 2, 2), 5.0)
        np.testing.assert_allclose(result.values, expected)

    def test_acceleration(self, valid_pose_dataset):
        """Test acceleration calculation."""
        data = valid_pose_dataset.pose_tracks
        result = kinematics.acceleration(data)
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
