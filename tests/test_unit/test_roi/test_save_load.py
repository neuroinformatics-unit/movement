"""Tests for saving and loading regions of interest to/from files."""

import json

import pytest

from movement.roi import LineOfInterest, PolygonOfInterest


class TestROISaveLoad:
    """Tests for ROI save/load functionality."""

    def test_save_and_load_polygon_roi(self, tmp_path, triangle):
        """Test round-trip save and load for PolygonOfInterest."""
        file_path = tmp_path / "triangle.json"

        # Save
        triangle.to_file(file_path)

        # Verify file exists and has correct content
        assert file_path.exists()
        data = json.loads(file_path.read_text())
        assert data["roi_type"] == "PolygonOfInterest"
        assert data["dimensions"] == 2
        assert data["name"] == "triangle"
        assert "geometry_wkt" in data

        # Load
        loaded = PolygonOfInterest.from_file(file_path)

        # Verify loaded ROI matches original
        assert loaded.name == triangle.name
        assert loaded.dimensions == triangle.dimensions
        assert loaded.region.equals(triangle.region)

    def test_save_and_load_line_roi(self, tmp_path, segment_of_y_equals_x):
        """Test round-trip save and load for LineOfInterest."""
        file_path = tmp_path / "line.json"

        # Save
        segment_of_y_equals_x.to_file(file_path)

        # Verify file exists
        assert file_path.exists()
        data = json.loads(file_path.read_text())
        assert data["roi_type"] == "LineOfInterest"
        assert data["dimensions"] == 1

        # Load
        loaded = LineOfInterest.from_file(file_path)

        # Verify loaded ROI matches original
        assert loaded.dimensions == segment_of_y_equals_x.dimensions
        assert loaded.region.equals(segment_of_y_equals_x.region)

    def test_save_and_load_polygon_with_hole(
        self, tmp_path, unit_square_with_hole
    ):
        """Test round-trip for polygon with interior holes."""
        file_path = tmp_path / "square_with_hole.json"

        # Save
        unit_square_with_hole.to_file(file_path)

        # Load
        loaded = PolygonOfInterest.from_file(file_path)

        # Verify holes are preserved
        assert loaded.region.equals(unit_square_with_hole.region)
        assert len(loaded.holes) == len(unit_square_with_hole.holes)

    def test_load_nonexistent_file_raises(self, tmp_path):
        """Test that loading a non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="ROI file not found"):
            PolygonOfInterest.from_file(tmp_path / "nonexistent.json")

    def test_save_with_none_name(self, tmp_path, triangle_pts):
        """Test saving an ROI with no name."""
        roi = PolygonOfInterest(triangle_pts)  # No name provided
        file_path = tmp_path / "unnamed.json"

        roi.to_file(file_path)
        loaded = PolygonOfInterest.from_file(file_path)

        # Name should be None in both
        assert loaded._name is None
