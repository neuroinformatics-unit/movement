"""Save pose tracking data from ``movement`` to various file formats."""

import json
import logging
import uuid
from collections.abc import Sequence
from pathlib import Path

import numpy as np

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Bboxes:
    """Container for bounding box coordinates in various formats.
    
    Parameters
    ----------
    coordinates : list or np.ndarray
        Array of bounding box coordinates
    format : str
        Coordinate format specification (e.g., 'xyxy', 'xywh')

    """

    def __init__(self, coordinates: list | np.ndarray, format: str):
        """Initialize with box coordinates and format."""
        self.coordinates = np.array(coordinates)
        self.format = format

    def convert(self, target_format: str, inplace: bool = False) -> "Bboxes":
        """Convert coordinates to target format.
        
        Parameters
        ----------
        target_format : str
            Desired output format ('xyxy' or 'xywh')
        inplace : bool, optional
            Whether to modify the current instance
            
        Returns
        -------
        Bboxes
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
            return Bboxes(new_coords, target_format)
            
        raise ValueError(
            f"Unsupported conversion: {self.format} -> {target_format}"
        )


def _validate_file_path(
    file_path: str | Path, 
    expected_suffix: Sequence[str]  # Changed to Sequence for indexable type
) -> Path:
    """Validate and normalize file paths."""
    path = Path(file_path).resolve()
    valid_suffixes = [s.lower() for s in expected_suffix]  # This is now indexable
    if path.suffix.lower() not in valid_suffixes:
        raise ValueError(
            f"Invalid file extension. Expected: {expected_suffix}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def to_via_tracks_file(
    boxes: Bboxes | dict[int, Bboxes],
    file_path: str | Path,
    video_metadata: dict | None = None,
) -> None:
    """Save bounding boxes to VIA-tracks format.
    
    Parameters
    ----------
    boxes : Bboxes or dict[int, Bboxes]
        Bounding boxes to export
    file_path : str or Path
        Output JSON file path
    video_metadata : dict, optional
        Video metadata including filename, size, etc.

    """
    file = _validate_file_path(file_path, [".json"])
    
    # Set default metadata
    video_metadata = video_metadata or {
        "filename": "unknown_video.mp4",
        "size": -1,
        "width": 0,
        "height": 0,
    }

    via_data = {
        "_via_settings": {
            "ui": {"file_content_align": "center"},
            "core": {"buffer_size": 18, "filepath": {}}
        },
        "_via_data_format_version": "2.0.10",
        "_via_image_id_list": [],
        "_via_attributes": {"region": {}, "file": {}},
        "_via_data": {"metadata": {}, "vid_list": {}, "cache": {}}
    }

    vid = str(uuid.uuid4())
    via_data["_via_data"]["vid_list"][vid] = {
        "fid_list": [],
        "filepath": video_metadata["filename"],
        "filetype": "video",
        "filesize": video_metadata["size"],
        "width": video_metadata["width"],
        "height": video_metadata["height"],
    }

    frame_dict = boxes if isinstance(boxes, dict) else {0: boxes}

    for frame_idx, frame_boxes in frame_dict.items():
        current_boxes = frame_boxes
        if frame_boxes.format != "xyxy":
            current_boxes = frame_boxes.convert("xyxy", inplace=False)

        fid = str(frame_idx)
        via_data["_via_data"]["vid_list"][vid]["fid_list"].append(fid)
        mid = f"{vid}_{fid}"
        
        via_data["_via_data"]["metadata"][mid] = {
            "vid": vid,
            "flg": 0,
            "z": [],
            "xy": [],
            "av": {},
        }

        for i, box in enumerate(current_boxes.coordinates):
            x1, y1, x2, y2 = box
            region = {
                "shape_attributes": {
                    "name": "rect",
                    "x": float(x1),
                    "y": float(y1),
                    "width": float(x2 - x1),
                    "height": float(y2 - y1)
                },
                "region_attributes": {"id": i}
            }
            via_data["_via_data"]["metadata"][mid]["xy"].append(region)

    with open(file, "w") as f:
        json.dump(via_data, f, indent=2)

    logger.info(f"Saved bounding boxes to VIA-tracks file: {file}")