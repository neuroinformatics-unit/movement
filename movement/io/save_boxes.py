"""Save pose tracking data from ``movement`` to various file formats."""

import json
import logging
import uuid
from pathlib import Path
from typing import Union

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _validate_file_path(
    file_path: str | Path, expected_suffix: list[str]
) -> Path:
    """Validate and normalize file paths."""
    path = Path(file_path).resolve()
    if path.suffix.lower() not in [s.lower() for s in expected_suffix]:
        raise ValueError(
            f"Invalid file extension. Expected: {expected_suffix}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def to_via_tracks_file(
    bboxes: Union["Bboxes", dict[int, "Bboxes"]],
    file_path: str | Path,
    video_metadata: dict | None = None,
) -> None:
    """Save bounding boxes to a VIA-tracks format file.

    Parameters
    ----------
    bboxes : Bboxes or dict[int, Bboxes]
        Bounding boxes to export. If dict, keys are frame indices.
    file_path : str or Path
        Path to save the VIA-tracks JSON file.
    video_metadata : dict, optional
        Video metadata including filename, size, width, height.
        Defaults to minimal metadata if None.

    Examples
    --------
    >>> from movement.io import save_poses
    >>> bboxes = Bboxes([[10, 20, 50, 60]], format="xyxy")
    >>> save_poses.to_via_tracks_file(
    ...     bboxes,
    ...     "output.json",
    ...     {"filename": "video.mp4", "width": 1280, "height": 720},
    ... )

    """
    file = _validate_file_path(file_path, expected_suffix=[".json"])

    # Create minimal metadata if not provided
    video_metadata = video_metadata or {
        "filename": "unknown_video.mp4",
        "size": -1,
        "width": 0,
        "height": 0,
    }

    # Initialize VIA-tracks structure
    via_data = {
        "_via_settings": {
            "ui": {"file_content_align": "center"},
            "core": {"buffer_size": 18, "filepath": {}},
        },
        "_via_data_format_version": "2.0.10",
        "_via_image_id_list": [],
        "_via_attributes": {"region": {}, "file": {}},
        "_via_data": {"metadata": {}, "vid_list": {}, "cache": {}},
    }

    # Create video ID
    vid = str(uuid.uuid4())
    via_data["_via_data"]["vid_list"][vid] = {
        "fid_list": [],
        "filepath": video_metadata["filename"],
        "filetype": "video",
        "filesize": video_metadata["size"],
        "width": video_metadata["width"],
        "height": video_metadata["height"],
    }

    # Process bboxes
    frame_dict = bboxes if isinstance(bboxes, dict) else {0: bboxes}

    for frame_idx, frame_bboxes in frame_dict.items():
        # Convert to xyxy format if needed
        current_bboxes = frame_bboxes
        if frame_bboxes.format != "xyxy":
            current_bboxes = frame_bboxes.convert("xyxy", inplace=False)

        # Add frame metadata
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

        # Add regions
        for i, bbox in enumerate(current_bboxes.bboxes):
            x1, y1, x2, y2 = bbox
            region = {
                "shape_attributes": {
                    "name": "rect",
                    "x": float(x1),
                    "y": float(y1),
                    "width": float(x2 - x1),
                    "height": float(y2 - y1),
                },
                "region_attributes": {"id": i},
            }
            via_data["_via_data"]["metadata"][mid]["xy"].append(region)

    # Save to file
    with open(file, "w") as f:
        json.dump(via_data, f, indent=2)

    logger.info(f"Saved bounding boxes to VIA-tracks file: {file}")
