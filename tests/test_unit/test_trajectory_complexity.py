"""Unit tests for the trajectory complexity module."""

import numpy as np
import pytest
import xarray as xr

from movement.trajectory_complexity import (
    compute_straightness_index,
    compute_sinuosity,
    compute_tortuosity,
    compute_angular_velocity,
    compute_directional_change,
)


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
    assert result.sel(keypoints="centroid", individual="test_subject").item() > 0.99


def test_straightness_index_zigzag(zigzag_trajectory):
    """Test that a zigzag path has straightness index less than 1."""
    result = compute_straightness_index(zigzag_trajectory)
    # Should be less than 1 for a zigzag path
    assert result.sel(keypoints="centroid", individual="test_subject").item() < 0.9


def test_straightness_index_circle(circular_trajectory):
    """Test that a circular path that returns to start has low straightness."""
    result = compute_straightness_index(circular_trajectory)
    # Should be very low for a circle that nearly returns to starting point
    assert result.sel(keypoints="centroid", individual="test_subject").item() < 0.2


def test_sinuosity_straight_line(straight_trajectory):
    """Test that a straight line has sinuosity close to 1."""
    result = compute_sinuosity(straight_trajectory, window_size=5)
    # Take the mean sinuosity over valid time points
    mean_sinuosity = result.sel(
        keypoints="centroid", individual="test_subject"
    ).mean(skipna=True)
    # Should be very close to 1 for a straight line
    assert mean_sinuosity.item() < 1.1


def test_sinuosity_zigzag(zigzag_trajectory):
    """Test that a zigzag path has sinuosity greater than 1."""
    result = compute_sinuosity(zigzag_trajectory, window_size=5)
    # Take the mean sinuosity over valid time points
    mean_sinuosity = result.sel(
        keypoints="centroid", individual="test_subject"
    ).mean(skipna=True)
    # Should be greater than 1 for a zigzag path
    assert mean_sinuosity.item() > 1.1


def test_angular_velocity_straight_line(straight_trajectory):
    """Test that a straight line has angular velocity close to 0."""
    result = compute_angular_velocity(straight_trajectory)
    # All angular velocities should be close to zero for a straight line
    max_ang_vel = np.nanmax(
        result.sel(keypoints="centroid", individual="test_subject").values
    )
    assert max_ang_vel < 1e-10


def test_angular_velocity_zigzag(zigzag_trajectory):
    """Test that a zigzag path has non-zero angular velocity."""
    result = compute_angular_velocity(zigzag_trajectory)
    # Should have some non-zero angular velocities
    max_ang_vel = np.nanmax(
        result.sel(keypoints="centroid", individual="test_subject").values
    )
    assert max_ang_vel > 0.1


def test_tortuosity_angular_variance_straight(straight_trajectory):
    """Test that a straight line has low angular variance tortuosity."""
    result = compute_tortuosity(
        straight_trajectory, method="angular_variance"
    )
    # Angular variance should be very close to 0 for a straight line
    assert result.sel(keypoints="centroid", individual="test_subject").item() < 0.1


def test_tortuosity_angular_variance_zigzag(zigzag_trajectory):
    """Test that a zigzag path has higher angular variance tortuosity."""
    result = compute_tortuosity(
        zigzag_trajectory, method="angular_variance"
    )
    # Angular variance should be higher for a zigzag path
    assert result.sel(keypoints="centroid", individual="test_subject").item() > 0.1


def test_tortuosity_fractal_straight(straight_trajectory):
    """Test that a straight line has fractal dimension close to 1."""
    result = compute_tortuosity(
        straight_trajectory, method="fractal"
    )
    # Fractal dimension should be close to 1 for a straight line
    tort = result.sel(keypoints="centroid", individual="test_subject").item()
    assert 0.9 < tort < 1.1


def test_tortuosity_fractal_zigzag(zigzag_trajectory):
    """Test that a zigzag path has fractal dimension greater than 1."""
    result = compute_tortuosity(
        zigzag_trajectory, method="fractal"
    )
    # Fractal dimension should be greater than 1 for a zigzag path
    tort = result.sel(keypoints="centroid", individual="test_subject").item()
    assert tort > 1.1


def test_directional_change_straight(straight_trajectory):
    """Test that a straight line has directional change close to 0."""
    result = compute_directional_change(straight_trajectory, window_size=5)
    # Sum of angular changes should be close to 0 for a straight line
    mean_dir_change = result.sel(
        keypoints="centroid", individual="test_subject"
    ).mean(skipna=True)
    assert mean_dir_change.item() < 1e-10


def test_directional_change_zigzag(zigzag_trajectory):
    """Test that a zigzag path has non-zero directional change."""
    result = compute_directional_change(zigzag_trajectory, window_size=5)
    # Sum of angular changes should be non-zero for a zigzag path
    mean_dir_change = result.sel(
        keypoints="centroid", individual="test_subject"
    ).mean(skipna=True)
    assert mean_dir_change.item() > 0.1 