"""Test suite for the save_bboxes module."""

import ast

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from movement.io import save_bboxes


@pytest.fixture
def create_valid_dataset():
    """Create a valid movement dataset for testing."""
    n_frames = 5
    n_space = 2
    n_individuals = 2

    # Create random data
    rng = np.random.default_rng(seed=42)
    position = rng.random((n_frames, n_space, n_individuals))
    shape = (
        rng.random((n_frames, n_space, n_individuals)) * 100
    )  # reasonable bbox sizes
    confidence = rng.random((n_frames, n_individuals))

    # Create dataset
    ds = xr.Dataset(
        data_vars={
            "position": (["time", "space", "individuals"], position),
            "shape": (["time", "space", "individuals"], shape),
            "confidence": (["time", "individuals"], confidence),
        },
        coords={
            "time": np.arange(n_frames),
            "space": ["x", "y"],
            "individuals": ["id_0", "id_1"],
        },
        attrs={
            "fps": 30.0,
            "source_software": "VIA-tracks",
        },
    )
    return ds


def test_to_via_tracks_file(tmp_path, create_valid_dataset):
    """Test saving a dataset to VIA-tracks format."""
    ds = create_valid_dataset
    output_file = tmp_path / "output.csv"

    # Save the dataset
    save_bboxes.to_via_tracks_file(ds, output_file)

    # Read the saved file
    df = pd.read_csv(output_file)

    # Check required columns exist
    required_columns = [
        "frame_filename",
        "region_id",
        "region_shape_attributes",
        "region_attributes",
    ]
    assert all(col in df.columns for col in required_columns)

    # Check number of rows (should be n_frames * n_individuals)
    assert len(df) == ds.sizes["time"] * ds.sizes["individuals"]

    # Check frame filenames
    expected_filenames = [
        f"frame_{i:06d}.jpg" for i in range(ds.sizes["time"])
    ]
    assert all(df["frame_filename"].unique() == expected_filenames)

    # Check region IDs
    assert all(df["region_id"].unique() == [0, 1])

    # Check shape attributes
    for _, row in df.iterrows():
        shape_attrs = ast.literal_eval(row["region_shape_attributes"])
        assert shape_attrs["name"] == "rect"
        assert all(key in shape_attrs for key in ["x", "y", "width", "height"])

        # Check that coordinates are converted from center to top-left
        frame_idx = int(row["frame_filename"].split("_")[1].split(".")[0])
        individual_idx = row["region_id"]

        # Get original center coordinates
        x_center = ds.position[frame_idx, 0, individual_idx].item()
        y_center = ds.position[frame_idx, 1, individual_idx].item()
        width = ds.shape[frame_idx, 0, individual_idx].item()
        height = ds.shape[frame_idx, 1, individual_idx].item()

        # Check conversion to top-left coordinates
        assert shape_attrs["x"] == x_center - width / 2
        assert shape_attrs["y"] == y_center - height / 2
        assert shape_attrs["width"] == width
        assert shape_attrs["height"] == height

    # Check region attributes
    for _, row in df.iterrows():
        region_attrs = ast.literal_eval(row["region_attributes"])
        assert "confidence" in region_attrs
        frame_idx = int(row["frame_filename"].split("_")[1].split(".")[0])
        individual_idx = row["region_id"]
        assert (
            region_attrs["confidence"]
            == ds.confidence[frame_idx, individual_idx].item()
        )


def test_to_via_tracks_file_invalid_file_extension(
    tmp_path, create_valid_dataset
):
    """Test that saving with invalid file extension raises an error."""
    ds = create_valid_dataset
    output_file = tmp_path / "output.txt"

    with pytest.raises(ValueError, match="Expected file extension"):
        save_bboxes.to_via_tracks_file(ds, output_file)


def test_to_via_tracks_file_invalid_dataset():
    """Test that saving an invalid dataset raises an error."""
    # Create an invalid dataset (missing required variables)
    ds = xr.Dataset(
        data_vars={
            "position": (
                ["time", "space", "individuals"],
                np.random.rand(5, 2, 2),
            ),
        },
        coords={
            "time": np.arange(5),
            "space": ["x", "y"],
            "individuals": ["id_0", "id_1"],
        },
    )

    with pytest.raises(ValueError):
        save_bboxes.to_via_tracks_file(ds, "output.csv")
