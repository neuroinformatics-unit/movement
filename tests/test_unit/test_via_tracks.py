"""Unit tests for the VIA-tracks file export functionality."""

import json
import os
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from movement.io.save_boxes import to_via_tracks_file


class TestVIATracksExport(unittest.TestCase):
    """Test suite for exporting bounding boxes to VIA-tracks CSV format."""

    def setUp(self):
        """Set up test data with a sample bounding boxes dataset."""
        # Create a sample dataset for testing
        n_frames = 5
        n_individuals = 2

        # Create sample position and shape data
        self.dataset = xr.Dataset(
            {
                "position": (
                    ("time", "space", "individuals"),
                    np.random.rand(n_frames, 2, n_individuals),
                ),
                "shape": (
                    ("time", "space", "individuals"),
                    np.random.rand(n_frames, 2, n_individuals),
                ),
                "confidence": (
                    ("time", "individuals"),
                    np.ones((n_frames, n_individuals)),  # confidence scores
                ),
            },
            coords={
                "time": np.arange(n_frames),
                "space": ["x", "y"],
                "individuals": [f"id_{i}" for i in range(n_individuals)],
            },
        )

    def tearDown(self):
        """Clean up temporary files after each test."""
        # Clean up any temporary files that might have been left behind
        for file in os.listdir(tempfile.gettempdir()):
            if file.endswith(".csv") and file.startswith("tmp"):
                try:
                    file_path = os.path.join(tempfile.gettempdir(), file)
                    # Try to delete the file with a small delay
                    for _ in range(3):  # Try up to 3 times
                        try:
                            os.unlink(file_path)
                            break
                        except PermissionError:
                            time.sleep(0.1)  # Wait 0.1 seconds before retrying
                except OSError:
                    pass

    def test_invalid_dataset_type(self):
        """Test that invalid dataset types raise TypeError."""
        with self.assertRaises(TypeError):
            to_via_tracks_file("not a dataset", "test.csv")

    def test_missing_required_variables(self):
        """Test that missing required variables raise ValueError."""
        # Create dataset without confidence variable
        invalid_ds = xr.Dataset(
            {
                "position": (
                    ("time", "space", "individuals"),
                    np.random.rand(5, 2, 2),
                ),
                "shape": (
                    ("time", "space", "individuals"),
                    np.random.rand(5, 2, 2),
                ),
            },
            coords={
                "time": np.arange(5),
                "space": ["x", "y"],
                "individuals": ["id_0", "id_1"],
            },
        )
        with self.assertRaises(ValueError) as cm:
            to_via_tracks_file(invalid_ds, "test.csv")
        self.assertIn("Missing required data variables", str(cm.exception))

    def test_missing_required_dimensions(self):
        """Test that missing required dimensions raise ValueError."""
        # Create dataset without 'individuals' dimension
        invalid_ds = xr.Dataset(
            {
                "position": (
                    ("time", "space"),
                    np.random.rand(5, 2),
                ),
                "shape": (
                    ("time", "space"),
                    np.random.rand(5, 2),
                ),
                "confidence": (
                    ("time",),
                    np.ones(5),
                ),
            },
            coords={
                "time": np.arange(5),
                "space": ["x", "y"],
            },
        )
        with self.assertRaises(ValueError) as cm:
            to_via_tracks_file(invalid_ds, "test.csv")
        self.assertIn("Missing required dimensions", str(cm.exception))

    def test_invalid_file_extension(self):
        """Test that invalid file extensions raise ValueError."""
        with self.assertRaises(ValueError):
            to_via_tracks_file(self.dataset, "test.txt")

    def test_empty_dataset(self):
        """Test handling of empty dataset."""
        empty_ds = xr.Dataset(
            {
                "position": (
                    ("time", "space", "individuals"),
                    np.zeros((0, 2, 0)),
                ),
                "shape": (
                    ("time", "space", "individuals"),
                    np.zeros((0, 2, 0)),
                ),
                "confidence": (
                    ("time", "individuals"),
                    np.zeros((0, 0)),
                ),
            },
            coords={
                "time": np.array([], dtype=int),
                "space": ["x", "y"],
                "individuals": [],
            },
        )
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp_path = tmp.name
            tmp.close()  # Close the file handle immediately

        # Ensure the file doesn't exist
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except PermissionError:
                time.sleep(0.1)
                os.unlink(tmp_path)

        output_path = to_via_tracks_file(empty_ds, tmp_path)
        df = pd.read_csv(output_path)
        self.assertEqual(len(df), 0)

        # Clean up
        try:
            os.unlink(output_path)
        except PermissionError:
            time.sleep(0.1)
            os.unlink(output_path)

    def test_all_nan_values(self):
        """Test handling of dataset with all NaN values."""
        nan_ds = xr.Dataset(
            {
                "position": (
                    ("time", "space", "individuals"),
                    np.full((5, 2, 2), np.nan),
                ),
                "shape": (
                    ("time", "space", "individuals"),
                    np.full((5, 2, 2), np.nan),
                ),
                "confidence": (
                    ("time", "individuals"),
                    np.full((5, 2), np.nan),
                ),
            },
            coords={
                "time": np.arange(5),
                "space": ["x", "y"],
                "individuals": ["id_0", "id_1"],
            },
        )
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp_path = tmp.name
            tmp.close()  # Close the file handle immediately

        # Ensure the file doesn't exist
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except PermissionError:
                time.sleep(0.1)
                os.unlink(tmp_path)

        output_path = to_via_tracks_file(nan_ds, tmp_path)
        df = pd.read_csv(output_path)
        self.assertEqual(len(df), 0)  # Should skip all rows with NaN values

        # Clean up
        try:
            os.unlink(output_path)
        except PermissionError:
            time.sleep(0.1)
            os.unlink(output_path)

    def test_file_creation(self):
        """Test that the VIA-tracks CSV file is created successfully."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp_path = tmp.name

        # Ensure the file doesn't exist
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except PermissionError:
                time.sleep(0.1)  # Wait a bit and try again
                os.unlink(tmp_path)

        output_path = to_via_tracks_file(self.dataset, tmp_path)
        self.assertTrue(os.path.exists(output_path))

        # Close any open file handles and delete
        try:
            os.unlink(output_path)
        except PermissionError:
            time.sleep(0.1)
            os.unlink(output_path)

    def test_file_content(self):
        """Test that the VIA-tracks CSV file contains the correct data."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp_path = tmp.name

        # Ensure the file doesn't exist
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except PermissionError:
                time.sleep(0.1)
                os.unlink(tmp_path)

        output_path = to_via_tracks_file(
            self.dataset, tmp_path, video_id="test_video"
        )

        df = pd.read_csv(output_path)

        self.assertEqual(len(df), 10)  # 5 times * 2 individuals
        self.assertEqual(
            list(df.columns),
            [
                "filename",
                "file_size",
                "file_attributes",
                "region_count",
                "region_id",
                "region_shape_attributes",
                "region_attributes",
            ],
        )

        # Check a sample row
        sample_row = df.iloc[0]
        self.assertTrue(sample_row["filename"].startswith("test_video_"))
        self.assertEqual(sample_row["file_size"], 0)
        self.assertEqual(sample_row["file_attributes"], "{}")
        self.assertEqual(sample_row["region_count"], 1)
        self.assertEqual(sample_row["region_id"], 0)

        # Check region_shape_attributes
        shape_attrs = json.loads(sample_row["region_shape_attributes"])
        self.assertEqual(shape_attrs["name"], "rect")
        self.assertIn("x", shape_attrs)
        self.assertIn("y", shape_attrs)
        self.assertIn("width", shape_attrs)
        self.assertIn("height", shape_attrs)

        # Check region_attributes
        region_attrs = json.loads(sample_row["region_attributes"])
        self.assertIn("track", region_attrs)
        self.assertIn(region_attrs["track"], self.dataset.individuals.values)

        # Close any open file handles and delete
        try:
            os.unlink(output_path)
        except PermissionError:
            time.sleep(0.1)
            os.unlink(output_path)

    def test_missing_data_handling(self):
        """Test that NaN values in the dataset are handled correctly."""
        # Create a dataset with some NaN values
        self.dataset["position"][0, 0, :] = (
            np.nan
        )  # Setting NaN for x-coordinate at time 0 for all individuals

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp_path = tmp.name

        # Ensure the file doesn't exist
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except PermissionError:
                time.sleep(0.1)
                os.unlink(tmp_path)

        output_path = to_via_tracks_file(self.dataset, tmp_path)

        df = pd.read_csv(output_path)

        # Let's calculate the expected number of rows:
        # Original dataset: 5 frames * 2 individuals = 10 rows
        # We set NaN for time 0, both individuals, so we lose 2 rows
        self.assertEqual(len(df), 8)

        # Close any open file handles and delete
        try:
            os.unlink(output_path)
        except PermissionError:
            time.sleep(0.1)
            os.unlink(output_path)

    def test_video_id_handling(self):
        """Test different video_id formats and handling."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp_path = tmp.name
            tmp.close()  # Close the file handle immediately

        # Ensure the file doesn't exist
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except PermissionError:
                time.sleep(0.1)
                os.unlink(tmp_path)

        # Test with custom video_id
        output_path = to_via_tracks_file(
            self.dataset, tmp_path, video_id="custom_video_123"
        )
        df = pd.read_csv(output_path)
        self.assertTrue(df["filename"].iloc[0].startswith("custom_video_123_"))

        # Clean up first file
        try:
            os.unlink(output_path)
        except PermissionError:
            time.sleep(0.1)
            os.unlink(output_path)

        # Test with default video_id (should use filename)
        output_path = to_via_tracks_file(self.dataset, tmp_path)
        df = pd.read_csv(output_path)
        expected_prefix = Path(tmp_path).stem
        self.assertTrue(
            df["filename"].iloc[0].startswith(f"{expected_prefix}_")
        )

        # Clean up second file
        try:
            os.unlink(output_path)
        except PermissionError:
            time.sleep(0.1)
            os.unlink(output_path)
