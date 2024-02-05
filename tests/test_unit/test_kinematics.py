import numpy as np
import pytest
import xarray as xr

from movement.analysis import kinematics


class TestKinematics:
    """Test suite for the kinematics module."""

    @pytest.fixture
    def expected_dataarray(self, valid_pose_dataset):
        """Return an xarray.DataArray with default values and
        the expected dimensions and coordinates."""
        return xr.DataArray(
            np.full((10, 2, 2, 2), [3.0, 4.0]),
            dims=valid_pose_dataset.dims,
            coords=valid_pose_dataset.coords,
        )

    @pytest.fixture
    def expected_dataset(self, valid_pose_dataset):
        """Return an xarray.Dataset with default `magnitude` and
        `direction` data variables, and the expected dimensions
        and coordinates."""
        dims = valid_pose_dataset.pose_tracks.dims[:-1]
        return xr.Dataset(
            data_vars={
                "magnitude": xr.DataArray(
                    np.full((10, 2, 2), 5.0),
                    dims=dims,
                ),
                "direction": xr.DataArray(
                    np.full((10, 2, 2), 0.92729522),
                    dims=dims,
                ),
            },
            coords={
                "time": valid_pose_dataset.time,
                "individuals": valid_pose_dataset.individuals,
                "keypoints": valid_pose_dataset.keypoints,
            },
        )

    def test_displacement(self, valid_pose_dataset, expected_dataarray):
        """Test displacement calculation."""
        result = kinematics.compute_displacement(
            valid_pose_dataset.pose_tracks
        )
        # Set the first displacement to zero
        expected_dataarray[0, :, :, :] = 0
        xr.testing.assert_allclose(result, expected_dataarray)

    def test_displacement_vector(self, valid_pose_dataset, expected_dataset):
        """Test displacement vector calculation."""
        result = kinematics.compute_displacement_vector(
            valid_pose_dataset.pose_tracks
        )
        # Set the first displacement to zero
        expected_dataset.magnitude[0, :, :] = 0
        expected_dataset.direction[0, :, :] = 0
        xr.testing.assert_allclose(result, expected_dataset)

    def test_velocity(self, valid_pose_dataset, expected_dataarray):
        """Test velocity calculation."""
        result = kinematics.compute_velocity(valid_pose_dataset.pose_tracks)
        xr.testing.assert_allclose(result, expected_dataarray)

    def test_velocity_vector(self, valid_pose_dataset, expected_dataset):
        """Test velocity vector calculation."""
        result = kinematics.compute_velocity_vector(
            valid_pose_dataset.pose_tracks
        )
        xr.testing.assert_allclose(result, expected_dataset)

    def test_acceleration(self, valid_pose_dataset, expected_dataarray):
        """Test acceleration calculation."""
        result = kinematics.compute_acceleration(
            valid_pose_dataset.pose_tracks
        )
        expected_dataarray[:] = 0
        xr.testing.assert_allclose(result, expected_dataarray)

    def test_acceleration_vector(self, valid_pose_dataset, expected_dataset):
        """Test acceleration vector calculation."""
        result = kinematics.compute_acceleration_vector(
            valid_pose_dataset.pose_tracks
        )
        expected_dataset.magnitude[:] = 0
        expected_dataset.direction[:] = 0
        xr.testing.assert_allclose(result, expected_dataset)

    def test_approximate_derivative_with_nonpositive_order(self):
        """Test that an error is raised when the order is non-positive."""
        data = np.arange(10)
        with pytest.raises(ValueError):
            kinematics.compute_approximate_derivative(data, order=0)
