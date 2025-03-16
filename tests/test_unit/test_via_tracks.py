import json
import logging

import numpy as np
import pytest

from movement.io.save_boxes import to_via_tracks_file


class Bboxes:
    """Mock Bboxes class for testing."""

    def __init__(self, bboxes, format):
        self.bboxes = np.array(bboxes)
        self.format = format

    def convert(self, target_format, inplace=False):
        if self.format == target_format:
            return self
        if self.format == "xywh" and target_format == "xyxy":
            converted = []
            for bbox in self.bboxes:
                x, y, w, h = bbox
                converted.append([x, y, x + w, y + h])
            new_bboxes = np.array(converted)
            if inplace:
                self.bboxes = new_bboxes
                self.format = target_format
                return self
            return Bboxes(new_bboxes, target_format)
        raise ValueError(
            f"Unsupported conversion: {self.format}->{target_format}"
        )


class TestVIATracksExport:
    """Test suite for VIA-tracks export functionality."""

    @pytest.fixture
    def sample_bboxes(self):
        return Bboxes([[10, 20, 50, 60]], format="xyxy")

    @pytest.fixture
    def video_metadata(self):
        return {
            "filename": "test_video.mp4",
            "width": 1280,
            "height": 720,
            "size": 1024000,
        }

    def test_basic_export(self, tmp_path, sample_bboxes, video_metadata):
        output_file = tmp_path / "output.json"
        to_via_tracks_file(sample_bboxes, output_file, video_metadata)

        assert output_file.exists()
        with open(output_file) as f:
            data = json.load(f)
            assert "_via_data" in data
            assert len(data["_via_data"]["vid_list"]) == 1

    def test_file_validation(self, tmp_path, sample_bboxes):
        # Test valid JSON
        valid_path = tmp_path / "valid.json"
        to_via_tracks_file(sample_bboxes, valid_path)

        # Test invalid extension
        invalid_path = tmp_path / "invalid.txt"
        with pytest.raises(ValueError):
            to_via_tracks_file(sample_bboxes, invalid_path)

    def test_metadata_handling(self, tmp_path, sample_bboxes):
        output_file = tmp_path / "output.json"
        to_via_tracks_file(sample_bboxes, output_file)

        with open(output_file) as f:
            data = json.load(f)
            vid = list(data["_via_data"]["vid_list"].keys())[0]
            assert data["_via_data"]["vid_list"][vid]["width"] == 0  # Default

    def test_logging(self, caplog, tmp_path, sample_bboxes):
        output_file = tmp_path / "output.json"
        with caplog.at_level(logging.INFO):
            to_via_tracks_file(sample_bboxes, output_file)
            assert "Saved bounding boxes" in caplog.text

    def test_format_conversion(self, tmp_path):
        output_file = tmp_path / "output.json"
        bboxes = Bboxes([[10, 20, 40, 40]], format="xywh")  # xywh input
        to_via_tracks_file(bboxes, output_file)

        with open(output_file) as f:
            data = json.load(f)
            region = data["_via_data"]["metadata"][
                list(data["_via_data"]["metadata"].keys())[0]
            ]["xy"][0]["shape_attributes"]
            assert region["width"] == 40.0  # 50-10 after conversion
