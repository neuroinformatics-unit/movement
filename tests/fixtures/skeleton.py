"""Skeleton-specific test fixtures."""

import numpy as np
import pytest
import xarray as xr

from movement.napari.skeleton.templates import MOUSE_TEMPLATE


def _generate_circular_motion(n_frames: int, spatial_dims: int) -> np.ndarray:
    """Generate circular motion trajectory."""
    angles = np.linspace(0, 4 * np.pi, n_frames)
    x = 100 + 50 * np.cos(angles)
    y = 100 + 50 * np.sin(angles)
    if spatial_dims == 3:
        z = 100 + 20 * np.sin(angles * 2)
        return np.stack([x, y, z], axis=0)
    return np.stack([x, y], axis=0)


def _generate_linear_motion(n_frames: int, spatial_dims: int) -> np.ndarray:
    """Generate linear motion trajectory."""
    x = np.linspace(0, 200, n_frames)
    y = np.linspace(0, 150, n_frames)
    if spatial_dims == 3:
        z = np.linspace(0, 100, n_frames)
        return np.stack([x, y, z], axis=0)
    return np.stack([x, y], axis=0)


def _generate_random_walk(
    n_frames: int, spatial_dims: int, rng: np.random.Generator
) -> np.ndarray:
    """Generate random walk trajectory."""
    steps = rng.normal(0, 5, (spatial_dims, n_frames))
    return np.cumsum(steps, axis=1) + 100


def _replicate_position_for_keypoints(
    position: np.ndarray,
    n_frames: int,
    spatial_dims: int,
    n_keypoints: int,
    n_individuals: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Replicate base position for all keypoints with random offsets."""
    position_full = np.zeros(
        (n_frames, spatial_dims, n_keypoints, n_individuals)
    )
    for kp_idx in range(n_keypoints):
        for ind_idx in range(n_individuals):
            offset = rng.normal(0, 2, (spatial_dims, n_frames))
            position_full[:, :, kp_idx, ind_idx] = (position + offset).T
    return position_full


@pytest.fixture
def synthetic_skeleton_dataset(rng):
    """Generate synthetic pose dataset with circular motion.

    Returns a factory function that creates datasets with configurable
    parameters.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random number generator fixture
    n_frames : int, optional
        Number of frames, by default 100
    n_keypoints : int, optional
        Number of keypoints, by default 5
    n_individuals : int, optional
        Number of individuals, by default 1
    spatial_dims : int, optional
        Spatial dimensions (2 or 3), by default 2
    motion_type : str, optional
        Type of motion: "circular", "random_walk", or "linear",
        by default "circular"

    Returns
    -------
    xr.Dataset
        Synthetic pose dataset

    """

    def _synthetic_skeleton_dataset(
        n_frames=100,
        n_keypoints=5,
        n_individuals=1,
        spatial_dims=2,
        motion_type="circular",
    ):
        time = np.arange(n_frames)

        # Generate base motion trajectory
        if motion_type == "circular":
            position = _generate_circular_motion(n_frames, spatial_dims)
        elif motion_type == "linear":
            position = _generate_linear_motion(n_frames, spatial_dims)
        else:  # random_walk
            position = _generate_random_walk(n_frames, spatial_dims, rng)

        # Replicate for all keypoints and individuals
        position_full = _replicate_position_for_keypoints(
            position, n_frames, spatial_dims, n_keypoints, n_individuals, rng
        )

        # Create confidence array
        confidence = np.full((n_frames, n_keypoints, n_individuals), 0.95)

        # Create coordinate names
        keypoint_names = [f"kp_{i}" for i in range(n_keypoints)]
        individual_names = [f"ind_{i}" for i in range(n_individuals)]
        space_names = ["x", "y"] if spatial_dims == 2 else ["x", "y", "z"]

        # Create dataset
        ds = xr.Dataset(
            data_vars={
                "position": xr.DataArray(
                    position_full,
                    dims=["time", "space", "keypoints", "individuals"],
                ),
                "confidence": xr.DataArray(
                    confidence,
                    dims=["time", "keypoints", "individuals"],
                ),
            },
            coords={
                "time": time,
                "space": space_names,
                "keypoints": keypoint_names,
                "individuals": individual_names,
            },
            attrs={
                "ds_type": "poses",
                "fps": 30,
                "time_unit": "frames",
                "source_software": "synthetic",
            },
        )

        return ds

    return _synthetic_skeleton_dataset


@pytest.fixture
def mouse_skeleton_config():
    """Return mouse skeleton template configuration."""
    return MOUSE_TEMPLATE


@pytest.fixture
def simple_skeleton_config():
    """Return a simple 3-keypoint triangle skeleton configuration."""
    return {
        "keypoints": ["kp_0", "kp_1", "kp_2"],
        "connections": [
            {
                "start": "kp_0",
                "end": "kp_1",
                "color": "#FF0000",
                "width": 2.0,
                "segment": "side1",
            },
            {
                "start": "kp_1",
                "end": "kp_2",
                "color": "#00FF00",
                "width": 2.0,
                "segment": "side2",
            },
            {
                "start": "kp_2",
                "end": "kp_0",
                "color": "#0000FF",
                "width": 2.0,
                "segment": "side3",
            },
        ],
    }


@pytest.fixture
def skeleton_dataset_with_nans(synthetic_skeleton_dataset):
    """Return a skeleton dataset with some NaN values."""
    ds = synthetic_skeleton_dataset(n_frames=50, n_keypoints=5)

    # Set some keypoints to NaN
    ds.position.loc[{"time": [10, 11, 12], "keypoints": "kp_1"}] = np.nan
    ds.position.loc[{"time": [25, 26], "keypoints": "kp_3"}] = np.nan

    return ds
