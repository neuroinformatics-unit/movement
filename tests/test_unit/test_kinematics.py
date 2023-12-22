import numpy as np
import pytest

from movement.analysis import kinematics


class TestKinematics:
    """Test suite for the kinematics module."""

    def test_compute_displacement(self, valid_pose_dataset):
        """Test the `approximate_derivative` function for
        calculating displacement."""
        # Select a single keypoint from a single individual
        data = valid_pose_dataset.pose_tracks.isel(keypoints=0, individuals=0)
        # Compute displacement
        displacement = kinematics.approximate_derivative(data)
        expected = np.pad([5.0] * 9, (1, 0), "constant")
        assert np.allclose(displacement, expected)

    @pytest.mark.parametrize("method", ["euclidean", "numerical"])
    def test_compute_velocity(self, valid_pose_dataset, method):
        """Test the `compute_velocity` function."""
        # Select a single keypoint from a single individual
        data = valid_pose_dataset.pose_tracks.isel(keypoints=0, individuals=0)
        # Compute velocity
        velocity = kinematics.compute_velocity(data, method=method)
        expected = np.pad([5.0] * 9, (1, 0), "constant")
        assert np.allclose(velocity, expected)

    def test_compute_acceleration(self, valid_pose_dataset):
        """Test the `approximate_derivative` function for
        calculating acceleration."""
        # Select a single keypoint from a single individual
        data = valid_pose_dataset.pose_tracks.isel(keypoints=0, individuals=0)
        # Compute acceleration
        acceleration = kinematics.approximate_derivative(data, order=2)
        assert np.allclose(acceleration, np.zeros(10))
