"""Load pose data from C3D (optical marker) files."""

from __future__ import annotations
from pathlib import Path
import numpy as np
import xarray as xr

# This line is the hook that connects your code to the main software
from movement.io.load import register_loader

@register_loader("C3D")
def from_c3d_file(
    file: str | Path,
    fps: float | None = None,
    **kwargs,
) -> xr.Dataset:
    """Load pose data from C3D files into movement's standard format."""
    
    try:
        import ezc3d
    except ImportError:
        raise ImportError("ezc3d is required. Install it with: pip install ezc3d")

    file_path = Path(file)
    if not file_path.exists():
        raise FileNotFoundError(f"C3D file not found: {file_path}")

    # 1. Load the file using the bulletproof engine
    try:
        c3d_file = ezc3d.c3d(str(file_path))
    except Exception as e:
        raise ValueError(f"Failed to load C3D file. Error: {e}")

    # Extract raw data: shape is [Axis(4), Markers, Time]
    raw_points = c3d_file['data']['points']
    
    if raw_points.size == 0:
        raise ValueError("C3D file contains no marker data")

    n_axes, n_markers, n_frames = raw_points.shape

    # 2. CHOP and SHUFFLE to match movement's requirements
    xyz_points = raw_points[0:3, :, :]
    
    # [Space, Markers, Time] -> [Time, Space, Markers]
    shuffled_points = np.transpose(xyz_points, (2, 0, 1))
    
    # STAPLE the Individuals dimension -> [Time, Space, Markers, Individuals]
    position_data = np.expand_dims(shuffled_points, axis=3)

    # 3. Extract correct metadata (Labels and FPS) directly from the file
    try:
        raw_labels = c3d_file['parameters']['POINT']['LABELS']['value']
        
        # In case the file only has one single label (returns a string instead of a list)
        if isinstance(raw_labels, str):
            raw_labels = [raw_labels]
            
        # The Bouncer: Force the labels list to perfectly match the data length (n_markers)
        # If the file gave us 48 labels, we chop off the extra 22. 
        marker_names = [str(label) for label in raw_labels][:n_markers]
        
        # If the file somehow gave us FEWER labels than data, we pad the rest with generic names
        while len(marker_names) < n_markers:
            marker_names.append(f"unlabeled_{len(marker_names)}")
            
    except KeyError:
        marker_names = [f"marker_{i}" for i in range(n_markers)]
        
    if fps is None:
        try:
            fps = c3d_file['header']['points']['frame_rate']
        except KeyError:
            fps = 100.0

    # 4. Package it into the official xarray format
    time_coords = np.arange(n_frames)
    space_coords = ["x", "y", "z"]
    individual_names = ["individual_0"]

    position = xr.DataArray(
        position_data,
        coords={
            "time": time_coords,
            "space": space_coords,
            "keypoints": marker_names,
            "individuals": individual_names,
        },
        dims=["time", "space", "keypoints", "individuals"],
        name="position",
    )

    # Create dummy confidence scores (since optical markers are highly accurate)
    confidence = xr.DataArray(
        np.ones((n_frames, n_markers, 1)),
        coords={
            "time": time_coords,
            "keypoints": marker_names,
            "individuals": individual_names,
        },
        dims=["time", "keypoints", "individuals"],
        name="confidence",
    )

    ds = xr.Dataset(
        {"position": position, "confidence": confidence},
        coords={
            "time": time_coords,
            "space": space_coords,
            "keypoints": marker_names,
            "individuals": individual_names,
        }
    )

    ds.attrs["source_software"] = "C3D"
    ds.attrs["fps"] = fps
    ds.attrs["ds_type"] = "poses"
    ds.attrs["file_path"] = str(file_path)

    return ds