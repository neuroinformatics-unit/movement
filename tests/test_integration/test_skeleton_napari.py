"""Integration tests for skeleton visualization with napari."""

import numpy as np
import pytest

try:
    import napari

    from movement.napari.skeleton import (
        SkeletonState,
        add_skeleton_layer,
    )
except ImportError:
    pytest.skip(
        "napari not installed - skipping integration tests",
        allow_module_level=True,
    )


def test_add_skeleton_layer_with_template(synthetic_skeleton_dataset):
    """Test adding skeleton layer using a template."""
    # Create dataset with mouse keypoints
    # Note: simple_skeleton_config uses kp_0, kp_1, kp_2
    ds = synthetic_skeleton_dataset(n_frames=20, n_keypoints=3)

    viewer = napari.Viewer(show=False)

    try:
        # Add skeleton layer with simple config
        layer = add_skeleton_layer(
            viewer,
            ds,
            connections={
                "keypoints": ["kp_0", "kp_1", "kp_2"],
                "connections": [
                    {
                        "start": "kp_0",
                        "end": "kp_1",
                        "color": "#FF0000",
                        "width": 2.0,
                        "segment": "seg1",
                    },
                ],
            },
            name="test_skeleton",
        )

        # Verify layer was added
        assert layer in viewer.layers
        assert layer.name == "test_skeleton"

        # Verify layer is Vectors type (renders as lines)
        assert isinstance(layer, napari.layers.Vectors)

        # Verify data shape is correct for napari (N, 2, D)
        assert layer.data.ndim == 3
        assert layer.data.shape[1] == 2  # [start, direction]
        assert layer.data.shape[2] == 3  # [t, y, x] for 2D+time

    finally:
        viewer.close()


def test_skeleton_layer_renders_as_lines_not_shapes(
    synthetic_skeleton_dataset, simple_skeleton_config
):
    """Test that skeleton renders as Vectors layer (lines) not Shapes layer."""
    ds = synthetic_skeleton_dataset(n_frames=10, n_keypoints=3)

    viewer = napari.Viewer(show=False)

    try:
        layer = add_skeleton_layer(
            viewer, ds, connections=simple_skeleton_config
        )

        # CRITICAL: Must be Vectors layer for line rendering
        assert type(layer).__name__ == "Vectors"
        assert not isinstance(layer, napari.layers.Shapes)

    finally:
        viewer.close()


def test_skeleton_layer_with_nan_keypoints(
    skeleton_dataset_with_nans, simple_skeleton_config
):
    """Test skeleton layer handles NaN keypoints correctly."""
    ds = skeleton_dataset_with_nans

    viewer = napari.Viewer(show=False)

    try:
        layer = add_skeleton_layer(
            viewer, ds, connections=simple_skeleton_config
        )

        # Should successfully create layer despite NaN values
        assert layer is not None

        # No NaN values in the rendered vectors
        assert not np.any(np.isnan(layer.data))

    finally:
        viewer.close()


def test_add_skeleton_layer_invalid_dataset():
    """Test that invalid dataset raises appropriate error."""
    import xarray as xr

    # Create non-poses dataset
    ds = xr.Dataset(attrs={"ds_type": "bboxes"})

    viewer = napari.Viewer(show=False)

    try:
        with pytest.raises(ValueError, match="poses dataset"):
            add_skeleton_layer(viewer, ds, connections={})
    finally:
        viewer.close()


def test_add_skeleton_layer_invalid_connections(synthetic_skeleton_dataset):
    """Test that invalid connections raise appropriate error."""
    ds = synthetic_skeleton_dataset(n_keypoints=3)

    viewer = napari.Viewer(show=False)

    try:
        # Config references non-existent keypoint
        invalid_config = {
            "keypoints": ["kp_0"],
            "connections": [
                {
                    "start": "kp_0",
                    "end": "nonexistent",
                    "color": "#FF0000",
                    "width": 2.0,
                    "segment": "",
                }
            ],
        }

        with pytest.raises(ValueError, match="not found"):
            add_skeleton_layer(viewer, ds, connections=invalid_config)
    finally:
        viewer.close()


def test_add_skeleton_layer_template_not_found(synthetic_skeleton_dataset):
    """Test that non-existent template name raises KeyError."""
    ds = synthetic_skeleton_dataset(n_keypoints=5)

    viewer = napari.Viewer(show=False)

    try:
        with pytest.raises(KeyError, match="Template"):
            add_skeleton_layer(viewer, ds, connections="nonexistent_template")
    finally:
        viewer.close()


def test_skeleton_state_persistence(
    synthetic_skeleton_dataset, simple_skeleton_config, tmp_path
):
    """Test that skeleton configuration persists in NetCDF."""
    ds = synthetic_skeleton_dataset(n_frames=10, n_keypoints=3)

    # Create skeleton state and embed in dataset
    state = SkeletonState.from_config(simple_skeleton_config, ds)
    ds_with_skeleton = state.to_dataset_attrs(ds)

    # Save to NetCDF using xarray's native save
    nc_path = tmp_path / "test_skeleton.nc"
    ds_with_skeleton.to_netcdf(nc_path)

    # Load from NetCDF
    import xarray as xr

    ds_loaded = xr.open_dataset(nc_path)

    # Extract skeleton state
    loaded_state = SkeletonState.from_dataset(ds_loaded)

    # Verify state was preserved
    assert loaded_state is not None
    assert len(loaded_state.connections) == len(state.connections)
    assert np.array_equal(loaded_state.edge_colors, state.edge_colors)
    assert np.array_equal(loaded_state.edge_widths, state.edge_widths)

    ds_loaded.close()


def test_skeleton_layer_multiple_individuals(
    synthetic_skeleton_dataset, simple_skeleton_config
):
    """Test skeleton visualization with multiple individuals."""
    ds = synthetic_skeleton_dataset(
        n_frames=10, n_keypoints=3, n_individuals=2
    )

    viewer = napari.Viewer(show=False)

    try:
        layer = add_skeleton_layer(
            viewer, ds, connections=simple_skeleton_config
        )

        # Should have vectors for both individuals
        # Number of vectors should be roughly:
        # n_frames * n_individuals * n_connections
        # (minus some for NaN handling)
        min_expected = 10 * 2 * 3 * 0.5  # At least half
        assert len(layer.data) >= min_expected

    finally:
        viewer.close()


def test_skeleton_layer_3d_dataset(
    synthetic_skeleton_dataset, simple_skeleton_config
):
    """Test skeleton visualization with 3D pose data."""
    ds = synthetic_skeleton_dataset(n_frames=10, n_keypoints=3, spatial_dims=3)

    viewer = napari.Viewer(show=False)

    try:
        layer = add_skeleton_layer(
            viewer, ds, connections=simple_skeleton_config
        )

        # For 3D+time, vectors should have shape (N, 2, 4)
        assert layer.data.shape[2] == 4  # [t, z, y, x]

    finally:
        viewer.close()
