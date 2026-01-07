"""Tests for PrecomputedRenderer."""

import numpy as np

from movement.napari.skeleton.config import config_to_arrays
from movement.napari.skeleton.renderers import PrecomputedRenderer


def test_precomputed_renderer_init(
    synthetic_skeleton_dataset, simple_skeleton_config
):
    """Test PrecomputedRenderer initialization."""
    ds = synthetic_skeleton_dataset(n_frames=50, n_keypoints=3)
    keypoint_names = list(ds.coords["keypoints"].values)
    connections, colors, widths, labels = config_to_arrays(
        simple_skeleton_config, keypoint_names
    )

    renderer = PrecomputedRenderer(
        dataset=ds,
        connections=connections,
        edge_colors=colors,
        edge_widths=widths,
    )

    assert renderer.name == "Precomputed"
    assert renderer.supports_3d is True
    assert renderer.requires_gpu is False
    assert renderer.vectors is None  # Not prepared yet


def test_precomputed_renderer_2d_vector_format(
    synthetic_skeleton_dataset, simple_skeleton_config
):
    """Test that 2D vectors have correct napari format (N, 2, 3)."""
    # Create 2D dataset
    ds = synthetic_skeleton_dataset(n_frames=10, n_keypoints=3, spatial_dims=2)
    keypoint_names = list(ds.coords["keypoints"].values)
    connections, colors, widths, labels = config_to_arrays(
        simple_skeleton_config, keypoint_names
    )

    renderer = PrecomputedRenderer(
        dataset=ds,
        connections=connections,
        edge_colors=colors,
        edge_widths=widths,
    )

    # Prepare (compute vectors)
    renderer.prepare()

    # Check vector format
    assert renderer.vectors is not None
    assert renderer.vectors.ndim == 3
    assert renderer.vectors.shape[1] == 2  # [start, direction]
    assert renderer.vectors.shape[2] == 3  # [t, y, x] for 2D+time

    # Check that we have reasonable number of vectors
    # 10 frames * 1 individual * 3 connections = 30 max vectors
    assert len(renderer.vectors) > 0
    assert len(renderer.vectors) <= 30


def test_precomputed_renderer_3d_vector_format(
    synthetic_skeleton_dataset, simple_skeleton_config
):
    """Test that 3D vectors have correct napari format (N, 2, 4)."""
    # Create 3D dataset
    ds = synthetic_skeleton_dataset(n_frames=10, n_keypoints=3, spatial_dims=3)
    keypoint_names = list(ds.coords["keypoints"].values)
    connections, colors, widths, labels = config_to_arrays(
        simple_skeleton_config, keypoint_names
    )

    renderer = PrecomputedRenderer(
        dataset=ds,
        connections=connections,
        edge_colors=colors,
        edge_widths=widths,
    )

    renderer.prepare()

    # Check vector format
    assert renderer.vectors.shape[1] == 2  # [start, direction]
    assert renderer.vectors.shape[2] == 4  # [t, z, y, x] for 3D+time


def test_precomputed_renderer_coordinate_order(
    synthetic_skeleton_dataset, simple_skeleton_config
):
    """Test that coordinates are in correct napari order [t, y, x]."""
    ds = synthetic_skeleton_dataset(n_frames=5, n_keypoints=3, spatial_dims=2)
    keypoint_names = list(ds.coords["keypoints"].values)
    connections, colors, widths, labels = config_to_arrays(
        simple_skeleton_config, keypoint_names
    )

    renderer = PrecomputedRenderer(
        dataset=ds,
        connections=connections,
        edge_colors=colors,
        edge_widths=widths,
    )

    renderer.prepare()

    # Take first vector
    if len(renderer.vectors) > 0:
        first_vector = renderer.vectors[0]
        start_pos = first_vector[0]

        # Check that time component is integer (frame index)
        assert start_pos[0] == int(start_pos[0])
        assert 0 <= start_pos[0] < 5  # Within frame range

        # Check that spatial coordinates are reasonable
        # (Our synthetic data is centered around 100)
        assert 0 < start_pos[1] < 200  # y coordinate
        assert 0 < start_pos[2] < 200  # x coordinate


def test_precomputed_renderer_direction_vector(
    synthetic_skeleton_dataset, simple_skeleton_config
):
    """Test that direction vectors are computed correctly."""
    ds = synthetic_skeleton_dataset(n_frames=5, n_keypoints=3, spatial_dims=2)
    keypoint_names = list(ds.coords["keypoints"].values)
    connections, colors, widths, labels = config_to_arrays(
        simple_skeleton_config, keypoint_names
    )

    renderer = PrecomputedRenderer(
        dataset=ds,
        connections=connections,
        edge_colors=colors,
        edge_widths=widths,
    )

    renderer.prepare()

    if len(renderer.vectors) > 0:
        first_vector = renderer.vectors[0]
        start_pos = first_vector[0]
        direction = first_vector[1]

        # Direction should have same dimensions as start
        assert direction.shape == start_pos.shape

        # Time component of direction should be 0
        # (vectors don't span across time)
        assert direction[0] == 0


def test_precomputed_renderer_nan_handling(
    skeleton_dataset_with_nans, simple_skeleton_config
):
    """Test that NaN keypoints are skipped correctly."""
    ds = skeleton_dataset_with_nans
    keypoint_names = list(ds.coords["keypoints"].values)
    connections, colors, widths, labels = config_to_arrays(
        simple_skeleton_config, keypoint_names
    )

    renderer = PrecomputedRenderer(
        dataset=ds,
        connections=connections,
        edge_colors=colors,
        edge_widths=widths,
    )

    renderer.prepare()

    # Verify no NaN values in output vectors
    assert not np.any(np.isnan(renderer.vectors))

    # Verify we have fewer vectors than total possible
    # (because some were skipped due to NaN)
    n_frames = ds.sizes["time"]
    n_individuals = ds.sizes["individuals"]
    n_connections = len(connections)
    max_vectors = n_frames * n_individuals * n_connections

    assert len(renderer.vectors) < max_vectors


def test_precomputed_renderer_cleanup(
    synthetic_skeleton_dataset, simple_skeleton_config
):
    """Test that cleanup properly releases memory."""
    ds = synthetic_skeleton_dataset(n_frames=10, n_keypoints=3)
    keypoint_names = list(ds.coords["keypoints"].values)
    connections, colors, widths, labels = config_to_arrays(
        simple_skeleton_config, keypoint_names
    )

    renderer = PrecomputedRenderer(
        dataset=ds,
        connections=connections,
        edge_colors=colors,
        edge_widths=widths,
    )

    renderer.prepare()
    assert renderer.vectors is not None

    renderer.cleanup()
    assert renderer.vectors is None


def test_precomputed_renderer_estimate_memory(
    synthetic_skeleton_dataset, simple_skeleton_config
):
    """Test memory estimation is reasonable."""
    ds = synthetic_skeleton_dataset(n_frames=100, n_keypoints=3)
    keypoint_names = list(ds.coords["keypoints"].values)
    connections, colors, widths, labels = config_to_arrays(
        simple_skeleton_config, keypoint_names
    )

    renderer = PrecomputedRenderer(
        dataset=ds,
        connections=connections,
        edge_colors=colors,
        edge_widths=widths,
    )

    estimated_mb = renderer.estimate_memory()

    # Should be a positive number
    assert estimated_mb > 0

    # Should be reasonable (not gigabytes for small dataset)
    assert estimated_mb < 100


def test_precomputed_renderer_get_info(
    synthetic_skeleton_dataset, simple_skeleton_config
):
    """Test get_info method returns correct information."""
    ds = synthetic_skeleton_dataset(n_frames=20, n_keypoints=3)
    keypoint_names = list(ds.coords["keypoints"].values)
    connections, colors, widths, labels = config_to_arrays(
        simple_skeleton_config, keypoint_names
    )

    renderer = PrecomputedRenderer(
        dataset=ds,
        connections=connections,
        edge_colors=colors,
        edge_widths=widths,
    )

    renderer.prepare()
    info = renderer.get_info()

    assert info["name"] == "Precomputed"
    assert info["n_frames"] == 20
    assert info["n_connections"] == 3
    assert "n_precomputed_vectors" in info
    assert "actual_memory_mb" in info


def test_precomputed_renderer_no_valid_vectors(simple_skeleton_config):
    """Test handling when all keypoints are NaN."""
    import xarray as xr

    # Create dataset with all NaN positions
    ds = xr.Dataset(
        data_vars={
            "position": xr.DataArray(
                np.full((5, 2, 3, 1), np.nan),
                dims=["time", "space", "keypoints", "individuals"],
            ),
            "confidence": xr.DataArray(
                np.zeros((5, 3, 1)),
                dims=["time", "keypoints", "individuals"],
            ),
        },
        coords={
            "time": np.arange(5),
            "space": ["x", "y"],
            "keypoints": ["kp_0", "kp_1", "kp_2"],
            "individuals": ["ind_0"],
        },
        attrs={"ds_type": "poses"},
    )

    keypoint_names = list(ds.coords["keypoints"].values)
    connections, colors, widths, labels = config_to_arrays(
        simple_skeleton_config, keypoint_names
    )

    renderer = PrecomputedRenderer(
        dataset=ds,
        connections=connections,
        edge_colors=colors,
        edge_widths=widths,
    )

    renderer.prepare()

    # Should return empty array with correct shape
    assert len(renderer.vectors) == 0
    assert renderer.vectors.shape == (0, 2, 3)
