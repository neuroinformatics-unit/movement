"""Unit tests for the trajectory complexity module."""

import numpy as np
import pytest
import xarray as xr

from movement.trajectory_complexity import compute_straightness_index


@pytest.fixture
def straight_trajectory():
    """Create a straight line trajectory."""
    position = np.zeros((20, 2, 1, 1))
    # x-coordinate increases linearly from 0 to 19
    position[:, 0, 0, 0] = np.arange(20)
    # y-coordinate stays at 0
    position[:, 1, 0, 0] = 0

    return xr.DataArray(
        position,
        dims=["time", "space", "keypoints", "individual"],
        coords={
            "time": np.arange(20) / 10,  # time in seconds
            "space": ["x", "y"],
            "keypoints": ["centroid"],
            "individual": ["test_subject"],
        },
    )


@pytest.fixture
def zigzag_trajectory():
    """Create a zigzag trajectory."""
    position = np.zeros((20, 2, 1, 1))
    # x-coordinate increases linearly from 0 to 19
    position[:, 0, 0, 0] = np.arange(20)
    # y-coordinate alternates between -1 and 1
    position[:, 1, 0, 0] = np.sin(np.arange(20) * np.pi / 2)

    return xr.DataArray(
        position,
        dims=["time", "space", "keypoints", "individual"],
        coords={
            "time": np.arange(20) / 10,  # time in seconds
            "space": ["x", "y"],
            "keypoints": ["centroid"],
            "individual": ["test_subject"],
        },
    )


@pytest.fixture
def circular_trajectory():
    """Create a circular trajectory."""
    position = np.zeros((20, 2, 1, 1))
    # x and y follow a circular path
    t = np.linspace(0, 2 * np.pi, 20)
    position[:, 0, 0, 0] = 5 * np.cos(t) + 10  # center at x=10
    position[:, 1, 0, 0] = 5 * np.sin(t) + 10  # center at y=10

    return xr.DataArray(
        position,
        dims=["time", "space", "keypoints", "individual"],
        coords={
            "time": np.arange(20) / 10,  # time in seconds
            "space": ["x", "y"],
            "keypoints": ["centroid"],
            "individual": ["test_subject"],
        },
    )


def test_straightness_index_straight_line(straight_trajectory):
    """Test that a straight line has straightness index close to 1."""
    result = compute_straightness_index(straight_trajectory)
    # Should be very close to 1 for a straight line
    assert (
        result.sel(keypoints="centroid", individual="test_subject").item()
        > 0.99
    )


def test_straightness_index_zigzag(zigzag_trajectory):
    """Test that a zigzag path has straightness index less than 1."""
    result = compute_straightness_index(zigzag_trajectory)
    # Should be less than 1 for a zigzag path
    assert (
        result.sel(keypoints="centroid", individual="test_subject").item()
        < 0.9
    )


def test_straightness_index_circle(circular_trajectory):
    """Test that a circular path that returns to start has low straightness."""
    result = compute_straightness_index(circular_trajectory)
    # Should be very low for a circle that nearly returns to starting point
    assert (
        result.sel(keypoints="centroid", individual="test_subject").item()
        < 0.2
    )


def test_straightness_index_with_time_range(straight_trajectory):
    """Test compute_straightness_index with explicit time range."""
    # Use only first half of trajectory
    result = compute_straightness_index(
        straight_trajectory, start=0.0, stop=1.0
    )
    # Should still be very close to 1 for partial straight line
    assert (
        result.sel(keypoints="centroid", individual="test_subject").item()
        > 0.99
    )
