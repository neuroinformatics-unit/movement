"""Tests for ROI serialization (to_file/from_file)."""

import json
from pathlib import Path

import pytest

from movement.roi import LineOfInterest, PolygonOfInterest


class TestSingleROISerialization:
    """Tests for saving and loading individual ROIs."""

    @pytest.mark.parametrize(
        ["roi_fixture", "expected_geom_type"],
        [
            pytest.param(
                "unit_square",
                "Polygon",
                id="Polygon",
            ),
            pytest.param(
                "unit_square_with_hole",
                "Polygon",
                id="Polygon with hole",
            ),
            pytest.param(
                "segment_of_y_equals_x",
                "LineString",
                id="LineString",
            ),
        ],
    )
    def test_to_file_creates_valid_geojson(
        self, roi_fixture, expected_geom_type, request, tmp_path
    ):
        """Test that to_file creates a valid GeoJSON file."""
        roi = request.getfixturevalue(roi_fixture)
        file_path = tmp_path / "roi.geojson"

        roi.to_file(file_path)

        assert file_path.exists()
        with open(file_path) as f:
            data = json.load(f)

        assert data["type"] == "Feature"
        assert data["geometry"]["type"] == expected_geom_type
        assert "properties" in data
        assert data["properties"]["name"] == roi.name

    @pytest.mark.parametrize(
        ["roi_fixture"],
        [
            pytest.param("unit_square", id="Polygon"),
            pytest.param("unit_square_with_hole", id="Polygon with hole"),
            pytest.param("segment_of_y_equals_x", id="LineString"),
        ],
    )
    def test_round_trip_preserves_geometry(
        self, roi_fixture, request, tmp_path
    ):
        """Test that save/load round-trip preserves geometry."""
        roi = request.getfixturevalue(roi_fixture)
        file_path = tmp_path / "roi.geojson"

        roi.to_file(file_path)

        if isinstance(roi, PolygonOfInterest):
            loaded_roi = PolygonOfInterest.from_file(file_path)
        else:
            loaded_roi = LineOfInterest.from_file(file_path)

        assert loaded_roi.region.equals(roi.region)

    @pytest.mark.parametrize(
        ["roi_fixture"],
        [
            pytest.param("unit_square", id="Polygon"),
            pytest.param("segment_of_y_equals_x", id="LineString"),
        ],
    )
    def test_round_trip_preserves_name(self, roi_fixture, request, tmp_path):
        """Test that save/load round-trip preserves name."""
        roi = request.getfixturevalue(roi_fixture)
        file_path = tmp_path / "roi.geojson"

        roi.to_file(file_path)

        if isinstance(roi, PolygonOfInterest):
            loaded_roi = PolygonOfInterest.from_file(file_path)
        else:
            loaded_roi = LineOfInterest.from_file(file_path)

        assert loaded_roi.name == roi.name

    def test_round_trip_linear_ring(self, tmp_path):
        """Test that a looped LineOfInterest round-trips correctly."""
        roi = LineOfInterest(
            [(0, 0), (1, 0), (1, 1), (0, 1)], loop=True, name="square_loop"
        )
        file_path = tmp_path / "loop.geojson"

        roi.to_file(file_path)
        loaded_roi = LineOfInterest.from_file(file_path)

        assert loaded_roi.is_closed
        assert loaded_roi.name == "square_loop"
        assert loaded_roi.region.equals(roi.region)

    def test_round_trip_polygon_with_holes(
        self, unit_square_pts, unit_square_hole, tmp_path
    ):
        """Test that a polygon with holes round-trips correctly."""
        roi = PolygonOfInterest(
            unit_square_pts, holes=[unit_square_hole], name="donut"
        )
        file_path = tmp_path / "donut.geojson"

        roi.to_file(file_path)
        loaded_roi = PolygonOfInterest.from_file(file_path)

        assert loaded_roi.name == "donut"
        assert loaded_roi.region.equals(roi.region)
        assert len(loaded_roi.holes) == 1

    def test_to_file_accepts_string_path(self, unit_square, tmp_path):
        """Test that to_file accepts string paths."""
        file_path = str(tmp_path / "roi.geojson")
        unit_square.to_file(file_path)
        assert Path(file_path).exists()

    def test_from_file_accepts_string_path(self, unit_square, tmp_path):
        """Test that from_file accepts string paths."""
        file_path = str(tmp_path / "roi.geojson")
        unit_square.to_file(file_path)
        loaded_roi = PolygonOfInterest.from_file(file_path)
        assert loaded_roi.region.equals(unit_square.region)

    @pytest.mark.parametrize(
        ["roi_class", "raw_geometry"],
        [
            pytest.param(
                LineOfInterest,
                '{"type": "LineString", "coordinates": [[0, 0], [1, 1]]}',
                id="LineString",
            ),
            pytest.param(
                PolygonOfInterest,
                '{"type": "Polygon", '
                '"coordinates": [[[0,0],[1,0],[1,1],[0,0]]]}',
                id="Polygon",
            ),
        ],
    )
    def test_from_file_raw_geometry(self, roi_class, raw_geometry, tmp_path):
        """Test from_file can load raw geometry (not wrapped in Feature)."""
        file_path = tmp_path / "raw.geojson"
        file_path.write_text(raw_geometry)

        roi = roi_class.from_file(file_path)
        assert roi is not None
        assert roi.name == "Un-named region"

    @pytest.mark.parametrize(
        ["roi_class", "raw_geometry", "expected_error"],
        [
            pytest.param(
                LineOfInterest,
                '{"type": "Polygon", '
                '"coordinates": [[[0,0],[1,0],[1,1],[0,0]]]}',
                "Expected LineString or LinearRing",
                id="Line from Polygon",
            ),
            pytest.param(
                PolygonOfInterest,
                '{"type": "LineString", "coordinates": [[0, 0], [1, 1]]}',
                "Expected Polygon geometry",
                id="Polygon from LineString",
            ),
        ],
    )
    def test_from_file_wrong_geometry_type(
        self, roi_class, raw_geometry, expected_error, tmp_path
    ):
        """Test that from_file raises error for wrong geometry type."""
        file_path = tmp_path / "wrong_type.geojson"
        file_path.write_text(raw_geometry)

        with pytest.raises(TypeError, match=expected_error):
            roi_class.from_file(file_path)


class TestROISequenceSerialization:
    """Tests for saving and loading collections of ROIs."""

    def test_save_rois_creates_feature_collection(
        self, unit_square, segment_of_y_equals_x, tmp_path
    ):
        """Test that save_rois creates a valid FeatureCollection."""
        from movement.roi import save_rois

        file_path = tmp_path / "rois.geojson"
        save_rois([unit_square, segment_of_y_equals_x], file_path)

        assert file_path.exists()
        with open(file_path) as f:
            data = json.load(f)

        assert data["type"] == "FeatureCollection"
        assert len(data["features"]) == 2

    def test_load_rois_returns_list(
        self, unit_square, segment_of_y_equals_x, tmp_path
    ):
        """Test that load_rois returns a list of ROI objects."""
        from movement.roi import load_rois, save_rois

        file_path = tmp_path / "rois.geojson"
        save_rois([unit_square, segment_of_y_equals_x], file_path)

        loaded_rois = load_rois(file_path)

        assert isinstance(loaded_rois, list)
        assert len(loaded_rois) == 2

    def test_collection_round_trip_preserves_types(
        self, unit_square, segment_of_y_equals_x, tmp_path
    ):
        """Test that collection round-trip preserves ROI types."""
        from movement.roi import load_rois, save_rois

        file_path = tmp_path / "rois.geojson"
        save_rois([unit_square, segment_of_y_equals_x], file_path)

        loaded_rois = load_rois(file_path)

        assert isinstance(loaded_rois[0], PolygonOfInterest)
        assert isinstance(loaded_rois[1], LineOfInterest)

    def test_collection_round_trip_preserves_geometry(
        self, unit_square, segment_of_y_equals_x, tmp_path
    ):
        """Test that collection round-trip preserves geometry."""
        from movement.roi import load_rois, save_rois

        original_rois = [unit_square, segment_of_y_equals_x]
        file_path = tmp_path / "rois.geojson"
        save_rois(original_rois, file_path)

        loaded_rois = load_rois(file_path)

        for original, loaded in zip(original_rois, loaded_rois, strict=True):
            assert loaded.region.equals(original.region)

    def test_collection_round_trip_preserves_names(
        self, unit_square, segment_of_y_equals_x, tmp_path
    ):
        """Test that collection round-trip preserves names."""
        from movement.roi import load_rois, save_rois

        original_rois = [unit_square, segment_of_y_equals_x]
        file_path = tmp_path / "rois.geojson"
        save_rois(original_rois, file_path)

        loaded_rois = load_rois(file_path)

        for original, loaded in zip(original_rois, loaded_rois, strict=True):
            assert loaded.name == original.name

    def test_save_empty_collection(self, tmp_path):
        """Test that saving an empty collection works."""
        from movement.roi import load_rois, save_rois

        file_path = tmp_path / "empty.geojson"
        save_rois([], file_path)

        loaded_rois = load_rois(file_path)
        assert loaded_rois == []

    def test_collection_with_mixed_roi_types(
        self,
        unit_square,
        unit_square_with_hole,
        segment_of_y_equals_x,
        tmp_path,
    ):
        """Test collection with various ROI types."""
        from movement.roi import load_rois, save_rois

        loop_line = LineOfInterest(
            [(0, 0), (1, 0), (1, 1)], loop=True, name="triangle_loop"
        )
        original_rois = [
            unit_square,
            unit_square_with_hole,
            segment_of_y_equals_x,
            loop_line,
        ]
        file_path = tmp_path / "mixed.geojson"

        save_rois(original_rois, file_path)
        loaded_rois = load_rois(file_path)

        assert len(loaded_rois) == 4
        assert isinstance(loaded_rois[0], PolygonOfInterest)
        assert isinstance(loaded_rois[1], PolygonOfInterest)
        assert isinstance(loaded_rois[2], LineOfInterest)
        assert isinstance(loaded_rois[3], LineOfInterest)
        assert loaded_rois[3].is_closed
