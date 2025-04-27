"""Integration tests for Parquet I/O in movement.io."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from movement.io import load_poses, save_poses


@pytest.fixture
def sample_dataset():
    """Create a sample xarray Dataset for testing."""
    rng = np.random.default_rng(seed=10)
    position_array = rng.random(
        (10, 2, 3, 2)
    )  # 10 frames, 2D, 3 keypoints, 2 individuals
    confidence_array = np.ones((10, 3, 2)) * 0.9
    return load_poses.from_numpy(
        position_array=position_array,
        confidence_array=confidence_array,
        individual_names=["ind1", "ind2"],
        keypoint_names=["nose", "tail", "spine"],
        fps=30,
        source_software="test",
    )


@pytest.fixture
def sample_tidy_df():
    """Create a sample tidy DataFrame for testing."""
    rng = np.random.default_rng(seed=12)
    data = {
        "frame": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        "track_id": ["ind1", "ind1", "ind1", "ind2", "ind2", "ind2"] * 2,
        "keypoint": ["nose", "tail", "spine"] * 4,
        "x": rng.random(12),
        "y": rng.random(12),
        "confidence": np.ones(12) * 0.9,
    }
    return pd.DataFrame(data)


def test_to_tidy_df(sample_dataset):
    """Test conversion of xarray Dataset to tidy DataFrame."""
    df = save_poses.to_tidy_df(sample_dataset)

    # Check columns
    expected_columns = {
        "frame",
        "track_id",
        "keypoint",
        "x",
        "y",
        "confidence",
    }
    assert set(df.columns) == expected_columns, (
        "Unexpected columns in tidy DataFrame"
    )

    # Check data types
    assert df["frame"].dtype == int, "Frame column should be integer"
    assert df["track_id"].dtype == object, (
        "Track_id column should be string/object"
    )
    assert df["keypoint"].dtype == object, (
        "Keypoint column should be string/object"
    )
    assert df["x"].dtype == float, "X column should be float"
    assert df["y"].dtype == float, "Y column should be float"
    assert df["confidence"].dtype == float, "Confidence column should be float"

    # Check shape
    expected_rows = (
        sample_dataset.sizes["time"]
        * sample_dataset.sizes["individuals"]
        * sample_dataset.sizes["keypoints"]
    )
    assert len(df) == expected_rows, (
        f"Expected {expected_rows} rows, got {len(df)}"
    )


def test_from_tidy_df(sample_tidy_df):
    """Test conversion of tidy DataFrame to xarray Dataset."""
    ds = load_poses.from_tidy_df(sample_tidy_df, fps=30)

    # Check dataset structure
    assert isinstance(ds, xr.Dataset), "Output should be an xarray Dataset"
    assert set(ds.dims) == {"time", "space", "keypoints", "individuals"}, (
        "Unexpected dimensions"
    )
    assert set(ds.data_vars) == {"position", "confidence"}, (
        "Unexpected data variables"
    )

    # Check coordinates
    assert ds.sizes["time"] == 2, "Expected 2 frames"
    assert ds.sizes["individuals"] == 2, "Expected 2 individuals"
    assert ds.sizes["keypoints"] == 3, "Expected 3 keypoints"
    assert ds.sizes["space"] == 2, "Expected 2D space"


def test_round_trip_dataframe(sample_dataset):
    """Test round-trip conversion: Dataset -> tidy DataFrame -> Dataset."""
    df = save_poses.to_tidy_df(sample_dataset)
    ds_roundtrip = load_poses.from_tidy_df(
        df, fps=sample_dataset.attrs.get("fps")
    )

    # Compare datasets
    xr.testing.assert_allclose(
        ds_roundtrip["position"], sample_dataset["position"]
    )
    xr.testing.assert_allclose(
        ds_roundtrip["confidence"], sample_dataset["confidence"]
    )
    assert ds_roundtrip.attrs["fps"] == sample_dataset.attrs["fps"], (
        "FPS metadata mismatch"
    )
    assert set(ds_roundtrip.coords["individuals"].values) == set(
        sample_dataset.coords["individuals"].values
    )
    assert set(ds_roundtrip.coords["keypoints"].values) == set(
        sample_dataset.coords["keypoints"].values
    )


def test_round_trip_parquet(sample_dataset, tmp_path):
    """Test round-trip conversion: Dataset -> Parquet -> Dataset."""
    file_path = tmp_path / "test.parquet"
    save_poses.to_animovement_file(sample_dataset, file_path)
    ds_roundtrip = load_poses.from_animovement_file(
        file_path, fps=sample_dataset.attrs.get("fps")
    )

    # Compare datasets
    xr.testing.assert_allclose(
        ds_roundtrip["position"], sample_dataset["position"]
    )
    xr.testing.assert_allclose(
        ds_roundtrip["confidence"], sample_dataset["confidence"]
    )
    assert ds_roundtrip.attrs["fps"] == sample_dataset.attrs["fps"], (
        "FPS metadata mismatch"
    )
    assert set(ds_roundtrip.coords["individuals"].values) == set(
        sample_dataset.coords["individuals"].values
    )
    assert set(ds_roundtrip.coords["keypoints"].values) == set(
        sample_dataset.coords["keypoints"].values
    )


def test_to_tidy_df_no_confidence():
    """Test to_tidy_df with a dataset lacking confidence scores."""
    rng = np.random.default_rng(seed=5)
    position_array = rng.random((5, 2, 2, 1))
    ds = load_poses.from_numpy(
        position_array=position_array,
        individual_names=["ind1"],
        keypoint_names=["nose", "tail"],
        fps=25,
    )
    df = save_poses.to_tidy_df(ds)

    # Check columns (no confidence)
    expected_columns = {"frame", "track_id", "keypoint", "x", "y"}
    assert set(df.columns) == expected_columns, (
        "Unexpected columns in tidy DataFrame"
    )
    assert len(df) == 5 * 1 * 2, "Incorrect number of rows"


def test_from_tidy_df_missing_columns(sample_tidy_df):
    """Test from_tidy_df with missing required columns."""
    invalid_df = sample_tidy_df.drop(columns=["x"])
    with pytest.raises(
        ValueError, match="DataFrame missing required columns: {'x'}"
    ):
        load_poses.from_tidy_df(invalid_df)


def test_from_animovement_file_invalid_extension(tmp_path):
    """Test from_animovement_file with incorrect file extension."""
    invalid_file = tmp_path / "test.csv"
    invalid_file.write_text("dummy")
    with pytest.raises(
        ValueError,
        match=r"Expected file with suffix\(es\) \['.parquet'\] "
        r"but got suffix .csv",
    ):
        load_poses.from_animovement_file(invalid_file)


def test_to_animovement_file_invalid_extension(tmp_path, sample_dataset):
    """Test to_animovement_file with incorrect file extension."""
    invalid_file = tmp_path / "test.csv"
    with pytest.raises(
        ValueError,
        match=r"Expected file with suffix\(es\) \['.parquet'\] "
        r"but got suffix .csv",
    ):
        save_poses.to_animovement_file(sample_dataset, invalid_file)


def test_empty_dataset():
    """Test handling of empty dataset."""
    empty_ds = load_poses.from_numpy(
        position_array=np.empty((0, 2, 0, 0)),
        confidence_array=np.empty((0, 0, 0)),
        individual_names=[],
        keypoint_names=[],
        fps=30,
    )
    df = save_poses.to_tidy_df(empty_ds)
    assert df.empty, "Tidy DataFrame should be empty for empty dataset"

    ds_roundtrip = load_poses.from_tidy_df(df, fps=30)
    assert ds_roundtrip.sizes["time"] == 0, (
        "Round-trip dataset should have zero frames"
    )
