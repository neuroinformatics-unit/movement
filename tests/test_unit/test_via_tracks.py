"""Unit tests for VIA-tracks export functionality."""

import json
import logging

import numpy as np
import pytest

from movement.io.save_boxes import to_via_tracks_file


class MockBboxes:
    """Test double for bounding box container class."""
    
    def __init__(self, coordinates: list | np.ndarray, format: str):
        """Initialize mock bounding boxes.
        
        Parameters
        ----------
        coordinates : list or np.ndarray
            Array of bounding box coordinates in specified format
        format : str
            Coordinate format specification (e.g., 'xyxy', 'xywh')

        """
        self.coordinates = np.array(coordinates)
        self.format = format

    def convert(self, target_format: str, inplace: bool = False) -> "MockBboxes":
        """Mock format conversion logic.
        
        Parameters
        ----------
        target_format : str
            Target coordinate format
        inplace : bool, optional
            Whether to modify the current instance
            
        Returns
        -------
        MockBboxes
            Converted bounding boxes

        """
        if self.format == target_format:
            return self
        if self.format == "xywh" and target_format == "xyxy":
            converted = []
            for box in self.coordinates:
                x, y, w, h = box
                converted.append([x, y, x + w, y + h])
            new_coords = np.array(converted)
            if inplace:
                self.coordinates = new_coords
                self.format = target_format
                return self
            return MockBboxes(new_coords, target_format)
        raise ValueError(
            f"Unsupported conversion: {self.format}->{target_format}"
        )

class TestVIATracksExport:
    """Test suite for VIA-tracks export functionality."""
    
    @pytest.fixture
    def sample_boxes_xyxy(self):
        """Provide sample boxes in xyxy format."""
        return MockBboxes([[10, 20, 50, 60]], format="xyxy")
    
    @pytest.fixture
    def sample_boxes_xywh(self):
        """Provide sample boxes in xywh format for conversion testing."""
        return MockBboxes([[10, 20, 40, 40]], format="xywh")
    
    @pytest.fixture
    def multi_frame_boxes(self):
        """Provide multi-frame box data as dictionary."""
        return {
            0: MockBboxes([[10, 20, 50, 60]], "xyxy"),
            1: MockBboxes([[30, 40, 70, 80]], "xyxy")
        }
    
    @pytest.fixture
    def video_metadata(self):
        """Provide standard video metadata for testing."""
        return {
            "filename": "test_video.mp4",
            "width": 1280,
            "height": 720,
            "size": 1024000,
        }

    def test_basic_export(self, tmp_path, sample_boxes_xyxy, video_metadata):
        """Verify successful export with valid inputs and metadata."""
        output_file = tmp_path / "output.json"
        to_via_tracks_file(sample_boxes_xyxy, output_file, video_metadata)
        
        assert output_file.exists()
        with open(output_file) as f:
            data = json.load(f)
            assert "_via_data" in data
            videos = data["_via_data"]["vid_list"]
            assert len(videos) == 1
            assert videos[list(videos.keys())[0]]["width"] == 1280

    def test_file_validation(self, tmp_path, sample_boxes_xyxy):
        """Test file path validation and error handling."""
        # Valid JSON path
        valid_path = tmp_path / "valid.json"
        to_via_tracks_file(sample_boxes_xyxy, valid_path)
        
        # Invalid extension
        invalid_path = tmp_path / "invalid.txt"
        with pytest.raises(ValueError) as exc_info:
            to_via_tracks_file(sample_boxes_xyxy, invalid_path)
        assert "Invalid file extension" in str(exc_info.value)

    def test_auto_metadata(self, tmp_path, sample_boxes_xyxy):
        """Verify default metadata generation when none is provided."""
        output_file = tmp_path / "output.json"
        to_via_tracks_file(sample_boxes_xyxy, output_file)
        
        with open(output_file) as f:
            data = json.load(f)
            vid = list(data["_via_data"]["vid_list"].keys())[0]
            assert data["_via_data"]["vid_list"][vid]["filepath"] == "unknown_video.mp4"

    def test_format_conversion(self, tmp_path, sample_boxes_xywh):
        """Test automatic conversion from xywh to xyxy format."""
        output_file = tmp_path / "converted.json"
        to_via_tracks_file(sample_boxes_xywh, output_file)
        
        with open(output_file) as f:
            data = json.load(f)
            region = data["_via_data"]["metadata"][
                list(data["_via_data"]["metadata"].keys())[0]
            ]["xy"][0]["shape_attributes"]
            assert abs(region["width"] - 40.0) < 1e-6  
            
    def test_multi_frame_export(self, tmp_path, multi_frame_boxes):
        """Verify correct handling of multi-frame input dictionaries."""
        output_file = tmp_path / "multi_frame.json"
        to_via_tracks_file(multi_frame_boxes, output_file)
        
        with open(output_file) as f:
            data = json.load(f)
            vid = list(data["_via_data"]["vid_list"].keys())[0]
            assert len(data["_via_data"]["vid_list"][vid]["fid_list"]) == 2

    def test_edge_cases(self, tmp_path):
        """Test handling of edge case values and empty inputs."""
        # Zero-size boxes
        output_file = tmp_path / "edge_cases.json"
        boxes = MockBboxes([[0, 0, 0, 0]], "xyxy")
        to_via_tracks_file(boxes, output_file)
        
        with open(output_file) as f:
            data = json.load(f)
            region = data["_via_data"]["metadata"][
                list(data["_via_data"]["metadata"].keys())[0]
            ]["xy"][0]["shape_attributes"]
            assert abs(region["width"] - 0.0) < 1e-6  

    def test_logging(self, caplog, tmp_path, sample_boxes_xyxy):
        """Verify proper logging of export operations."""
        output_file = tmp_path / "logging_test.json"
        with caplog.at_level(logging.INFO):
            to_via_tracks_file(sample_boxes_xyxy, output_file)
            assert "Saved bounding boxes" in caplog.text
            assert str(output_file) in caplog.text

    def test_error_handling(self, tmp_path):
        """Test proper error reporting for invalid inputs."""
        # Invalid box format
        invalid_boxes = MockBboxes([[10, 20, 50]], "invalid_format")
        with pytest.raises(ValueError):
            to_via_tracks_file(invalid_boxes, tmp_path / "test.json")