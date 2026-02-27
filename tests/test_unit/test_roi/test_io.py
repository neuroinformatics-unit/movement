"""Tests for IO of RoI collections."""

import json

from movement.roi import LineOfInterest, PolygonOfInterest


class TestROICollectionSerialization:
    """Tests for saving and loading collections of RoIs."""

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
        """Test that load_rois returns a list of RoI objects."""
        from movement.roi import load_rois, save_rois

        file_path = tmp_path / "rois.geojson"
        save_rois([unit_square, segment_of_y_equals_x], file_path)

        loaded_rois = load_rois(file_path)

        assert isinstance(loaded_rois, list)
        assert len(loaded_rois) == 2

    def test_collection_round_trip_preserves_types(
        self, unit_square, segment_of_y_equals_x, tmp_path
    ):
        """Test that collection round-trip preserves RoI types."""
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
        """Test collection with various RoI types."""
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
