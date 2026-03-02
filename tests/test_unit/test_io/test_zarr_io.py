"""Tests for zarr I/O functionality."""

import pytest
from xarray.testing import assert_equal

from movement.io import load_zarr, save_zarr


class TestSaveZarr:
    """Test zarr saving functionality."""

    def test_save_creates_zarr_store(self, valid_poses_dataset, tmp_path):
        """Test that saving creates a zarr directory."""
        zarr_path = tmp_path / "output.zarr"

        save_zarr.to_zarr(valid_poses_dataset, zarr_path)

        assert zarr_path.exists()
        assert zarr_path.is_dir()

    def test_save_overwrite_mode(self, valid_poses_dataset, tmp_path):
        """Test that mode='w' overwrites an existing store."""
        zarr_path = tmp_path / "overwrite.zarr"

        save_zarr.to_zarr(valid_poses_dataset, zarr_path)
        save_zarr.to_zarr(valid_poses_dataset, zarr_path, mode="w")

        loaded_ds = load_zarr.from_zarr(zarr_path, chunks=None)
        assert_equal(valid_poses_dataset, loaded_ds)

    def test_save_logs_info(self, valid_poses_dataset, tmp_path, caplog):
        """Test that saving logs the expected info messages."""
        import logging

        zarr_path = tmp_path / "log_test.zarr"

        with caplog.at_level(logging.INFO):
            save_zarr.to_zarr(valid_poses_dataset, zarr_path)

        assert str(zarr_path) in caplog.text


class TestLoadZarr:
    """Test zarr loading functionality."""

    def test_load_nonexistent_path_raises(self, tmp_path):
        """Test that loading a non-existent path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_zarr.from_zarr(tmp_path / "nonexistent.zarr")

    def test_load_file_instead_of_dir_raises(self, tmp_path):
        """Test that loading a file (not a directory) raises ValueError."""
        fake_file = tmp_path / "not_a_zarr.zarr"
        fake_file.touch()

        with pytest.raises(ValueError, match="Expected a directory"):
            load_zarr.from_zarr(fake_file)

    def test_load_eager_chunks_none(self, valid_poses_dataset, tmp_path):
        """Test that chunks=None loads data eagerly (no dask arrays)."""
        zarr_path = tmp_path / "eager.zarr"
        save_zarr.to_zarr(valid_poses_dataset, zarr_path)

        loaded_ds = load_zarr.from_zarr(zarr_path, chunks=None)

        # With chunks=None, data should be numpy arrays, not dask arrays
        import numpy as np

        for var in loaded_ds.data_vars:
            assert isinstance(loaded_ds[var].data, np.ndarray)

    def test_load_logs_info(self, valid_poses_dataset, tmp_path, caplog):
        """Test that loading logs the expected info messages."""
        import logging

        zarr_path = tmp_path / "log_load.zarr"
        save_zarr.to_zarr(valid_poses_dataset, zarr_path)

        with caplog.at_level(logging.INFO):
            load_zarr.from_zarr(zarr_path)

        assert str(zarr_path) in caplog.text


class TestZarrRoundTrip:
    """Test saving to and loading from zarr stores."""

    def test_round_trip_poses_dataset(self, valid_poses_dataset, tmp_path):
        """Test that a poses dataset survives a zarr round-trip."""
        zarr_path = tmp_path / "test_poses.zarr"

        save_zarr.to_zarr(valid_poses_dataset, zarr_path)
        loaded_ds = load_zarr.from_zarr(zarr_path, chunks=None)

        assert_equal(valid_poses_dataset, loaded_ds)

    def test_round_trip_bboxes_dataset(self, valid_bboxes_dataset, tmp_path):
        """Test that a bboxes dataset survives a zarr round-trip."""
        zarr_path = tmp_path / "test_bboxes.zarr"

        save_zarr.to_zarr(valid_bboxes_dataset, zarr_path)
        loaded_ds = load_zarr.from_zarr(zarr_path, chunks=None)

        assert_equal(valid_bboxes_dataset, loaded_ds)

    def test_round_trip_preserves_attrs(self, valid_poses_dataset, tmp_path):
        """Test that dataset attributes are preserved in zarr round-trip."""
        zarr_path = tmp_path / "test_attrs.zarr"

        save_zarr.to_zarr(valid_poses_dataset, zarr_path)
        loaded_ds = load_zarr.from_zarr(zarr_path, chunks=None)

        assert loaded_ds.attrs == valid_poses_dataset.attrs

    def test_round_trip_preserves_dimensions(
        self, valid_poses_dataset, tmp_path
    ):
        """Test that dataset dimensions are preserved in zarr round-trip."""
        zarr_path = tmp_path / "test_dims.zarr"

        save_zarr.to_zarr(valid_poses_dataset, zarr_path)
        loaded_ds = load_zarr.from_zarr(zarr_path, chunks=None)

        assert dict(loaded_ds.sizes) == dict(valid_poses_dataset.sizes)

    def test_round_trip_preserves_coordinates(
        self, valid_poses_dataset, tmp_path
    ):
        """Test that coordinate values are preserved in zarr round-trip."""
        zarr_path = tmp_path / "test_coords.zarr"

        save_zarr.to_zarr(valid_poses_dataset, zarr_path)
        loaded_ds = load_zarr.from_zarr(zarr_path, chunks=None)

        for coord in valid_poses_dataset.coords:
            assert_equal(
                valid_poses_dataset.coords[coord],
                loaded_ds.coords[coord],
            )

    def test_round_trip_string_path(self, valid_poses_dataset, tmp_path):
        """Test that a string path works the same as a pathlib.Path."""
        zarr_path = str(tmp_path / "test_str_path.zarr")

        save_zarr.to_zarr(valid_poses_dataset, zarr_path)
        loaded_ds = load_zarr.from_zarr(zarr_path, chunks=None)

        assert_equal(valid_poses_dataset, loaded_ds)
