import numpy as np
import pytest

from movement.analysis import kinematics


class TestKinematics:
    """Test suite for the kinematics module."""

    def test_distance(self, valid_pose_dataset):
        """Test distance calculation."""
        # Select a single keypoint from a single individual
        data = valid_pose_dataset.pose_tracks.isel(keypoints=0, individuals=0)
        result = kinematics.distance(data)
        expected = np.pad([5.0] * 9, (1, 0), "constant")
        assert np.allclose(result, expected)

    def test_displacement(self, valid_pose_dataset):
        """Test displacement calculation."""
        # Select a single keypoint from a single individual
        data = valid_pose_dataset.pose_tracks.isel(keypoints=0, individuals=0)
        result = kinematics.displacement(data)
        expected_magnitude = np.pad([5.0] * 9, (1, 0), "constant")
        expected_direction = np.concatenate(([0], np.full(9, 0.92729522)))
        expected = np.stack((expected_magnitude, expected_direction), axis=1)
        assert np.allclose(result, expected)

    def test_velocity(self, valid_pose_dataset):
        """Test velocity calculation."""
        # Select a single keypoint from a single individual
        data = valid_pose_dataset.pose_tracks.isel(keypoints=0, individuals=0)
        # Compute velocity
        result = kinematics.velocity(data)
        expected_magnitude = np.pad([5.0] * 9, (1, 0), "constant")
        expected_direction = np.concatenate(([0], np.full(9, 0.92729522)))
        expected = np.stack((expected_magnitude, expected_direction), axis=1)
        assert np.allclose(result, expected)

    def test_speed(self, valid_pose_dataset):
        """Test velocity calculation."""
        # Select a single keypoint from a single individual
        data = valid_pose_dataset.pose_tracks.isel(keypoints=0, individuals=0)
        result = kinematics.speed(data)
        expected = np.pad([5.0] * 9, (1, 0), "constant")
        assert np.allclose(result, expected)

    def test_acceleration(self, valid_pose_dataset):
        """Test acceleration calculation."""
        # Select a single keypoint from a single individual
        data = valid_pose_dataset.pose_tracks.isel(keypoints=0, individuals=0)
        result = kinematics.acceleration(data)
        assert np.allclose(result, np.zeros((10, 2)))

    def test_approximate_derivative_with_nonpositive_order(self):
        """Test that an error is raised when the order is non-positive."""
        data = np.arange(10)
        with pytest.raises(ValueError):
            kinematics.approximate_derivative(data, order=0)
