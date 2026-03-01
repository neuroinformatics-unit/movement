"""Tests for skeleton configuration module."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from movement.napari.skeleton.config import (
    config_to_arrays,
    connections_to_edge_indices,
    hex_to_rgba,
    load_yaml_config,
    rgba_to_hex,
    save_yaml_config,
    validate_config,
)


def test_hex_to_rgba():
    """Test hex color to RGBA conversion."""
    # Test with hash
    rgba = hex_to_rgba("#FF0000")
    assert rgba == (1.0, 0.0, 0.0, 1.0)

    # Test without hash
    rgba = hex_to_rgba("00FF00")
    assert rgba == (0.0, 1.0, 0.0, 1.0)

    # Test with alpha
    rgba = hex_to_rgba("#0000FF", alpha=0.5)
    assert rgba == (0.0, 0.0, 1.0, 0.5)


def test_hex_rgba_roundtrip():
    """Test that hex->rgba->hex conversion is lossless."""
    original = "#AABBCC"
    rgba = hex_to_rgba(original)
    result = rgba_to_hex(rgba)
    assert result == original


def test_connections_to_edge_indices(simple_skeleton_config):
    """Test conversion of connection names to indices."""
    keypoint_names = simple_skeleton_config["keypoints"]
    connections = simple_skeleton_config["connections"]

    edges = connections_to_edge_indices(connections, keypoint_names)

    assert len(edges) == 3
    assert edges[0] == (0, 1)  # kp_0 -> kp_1
    assert edges[1] == (1, 2)  # kp_1 -> kp_2
    assert edges[2] == (2, 0)  # kp_2 -> kp_0


def test_connections_to_edge_indices_invalid_keypoint():
    """Test that invalid keypoint names raise ValueError."""
    connections = [{"start": "invalid", "end": "kp_1"}]
    keypoint_names = ["kp_0", "kp_1", "kp_2"]

    with pytest.raises(ValueError, match="not found"):
        connections_to_edge_indices(connections, keypoint_names)


def test_config_to_arrays(simple_skeleton_config):
    """Test conversion of config dict to renderer arrays."""
    keypoint_names = simple_skeleton_config["keypoints"]

    connections, colors, widths, labels = config_to_arrays(
        simple_skeleton_config, keypoint_names
    )

    # Check connections
    assert len(connections) == 3
    assert connections[0] == (0, 1)

    # Check colors
    assert colors.shape == (3, 4)
    assert np.allclose(colors[0], (1.0, 0.0, 0.0, 1.0))  # Red

    # Check widths
    assert widths.shape == (3,)
    assert widths[0] == pytest.approx(2.0)

    # Check labels
    assert len(labels) == 3
    assert labels[0] == "side1"


def test_save_and_load_yaml_config(simple_skeleton_config):
    """Test saving and loading YAML configuration."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "test_config.yaml"

        # Save config
        save_yaml_config(simple_skeleton_config, config_path)
        assert config_path.exists()

        # Load config
        loaded_config = load_yaml_config(config_path)

        # Verify content
        assert (
            loaded_config["keypoints"] == simple_skeleton_config["keypoints"]
        )
        assert len(loaded_config["connections"]) == len(
            simple_skeleton_config["connections"]
        )


def test_load_yaml_config_nonexistent_file():
    """Test that loading non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_yaml_config("/nonexistent/path/config.yaml")


def test_validate_config_valid(
    simple_skeleton_config, synthetic_skeleton_dataset
):
    """Test validation of a valid configuration."""
    ds = synthetic_skeleton_dataset(n_keypoints=3)

    is_valid, errors = validate_config(simple_skeleton_config, ds)

    assert is_valid
    assert len(errors) == 0


def test_validate_config_missing_connections(synthetic_skeleton_dataset):
    """Test validation fails when connections key is missing."""
    ds = synthetic_skeleton_dataset(n_keypoints=3)
    invalid_config = {"keypoints": ["kp_0", "kp_1"]}

    is_valid, errors = validate_config(invalid_config, ds)

    assert not is_valid
    assert any("connections" in err for err in errors)


def test_validate_config_invalid_keypoint(
    simple_skeleton_config, synthetic_skeleton_dataset
):
    """Test validation fails when connection references invalid keypoint."""
    ds = synthetic_skeleton_dataset(n_keypoints=2)  # Only 2 keypoints

    # Config has connections to kp_2 which doesn't exist
    is_valid, errors = validate_config(simple_skeleton_config, ds)

    assert not is_valid
    assert any("not found" in err for err in errors)


def test_validate_config_no_keypoints_in_dataset(simple_skeleton_config):
    """Test validation fails when dataset has no keypoints coordinate."""
    import xarray as xr

    # Create dataset without keypoints
    ds = xr.Dataset(attrs={"ds_type": "poses"})

    is_valid, errors = validate_config(simple_skeleton_config, ds)

    assert not is_valid
    assert any("keypoints" in err.lower() for err in errors)
