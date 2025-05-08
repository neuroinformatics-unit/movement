"""Save bounding boxes data from ``movement`` to VIA-tracks CSV format."""

import csv
import json
from pathlib import Path

import numpy as np
import xarray as xr

from movement.io.save_poses import _validate_dataset, _validate_file_path
from movement.utils.logging import logger


def _write_single_via_row(
    frame: int,
    track_id: str,
    xy_coordinates: np.ndarray,
    wh_values: np.ndarray,
    video_id: str | None = None,
) -> list:
    """Return a list representing a single row for the VIA-tracks CSV file.

    Parameters
    ----------
    frame : int
        Frame number.
    track_id : str
        Track identifier.
    xy_coordinates : np.ndarray
        Position data (x, y).
    wh_values : np.ndarray
        Shape data (width, height).
    video_id : str | None, optional
        Video identifier, prepended to frame number when constructing filename.
        If None, nothing is prepended to the frame number.

    Returns
    -------
    list
        Row data in VIA-tracks format.

    """
    # Calculate top-left coordinates
    x_center, y_center = xy_coordinates
    width, height = wh_values
    x_top_left = x_center - width / 2
    y_top_left = y_center - height / 2

    # Define region shape attributes
    region_shape_attributes = json.dumps(
        {
            "name": "rect",
            "x": int(x_top_left),  # ----does it need to be int?
            "y": int(y_top_left),  # does it need to be int?
            "width": int(width),  # does it need to be int?
            "height": int(height),  # does it need to be int?
        }
    )

    # Define region attributes
    # TODO: include confidence score?
    # confidence = ds.confidence[frame_idx, individual_idx].item()
    region_attributes = json.dumps({"track": track_id})

    # Define filename
    filename_prefix = f"{f'{video_id}_' if video_id else ''}"

    # Define row data
    return [
        f"{filename_prefix}{frame:06d}.jpg",  # filename
        0,  # file_size (placeholder)
        "{}",  # file_attributes (empty JSON object)
        1,  # region_count ---set to 0?
        0,  # region_id
        region_shape_attributes,
        region_attributes,
    ]


def to_via_tracks_file(
    ds: xr.Dataset,
    file_path: str | Path,
    video_id: str | None = None,
) -> Path:
    """Save a movement bounding boxes dataset to a VIA-tracks CSV file.

    Parameters
    ----------
    ds : xarray.Dataset
        The movement bounding boxes dataset to export.
    file_path : str or pathlib.Path
        Path where the VIA-tracks CSV file will be saved.
    video_id : str, optional
        Video identifier to prepend to frame number when constructing the
        image filename. If None, nothing will be prepended.

    Returns
    -------
    pathlib.Path
        Path to the saved file.

    Examples
    --------
    >>> from movement.io import save_boxes, load_boxes
    >>> ds = load_boxes.from_via_tracks_file("/path/to/file.csv")
    >>> save_boxes.to_via_tracks_file(ds, "/path/to/output.csv")

    """
    # Validate file path and dataset
    file = _validate_file_path(file_path, expected_suffix=[".csv"])
    _validate_dataset(ds)

    with open(file.path, "w", newline="") as f:
        writer = csv.writer(f)

        # Write header
        writer.writerow(
            [
                "filename",
                "file_size",
                "file_attributes",
                "region_count",
                "region_id",
                "region_shape_attributes",
                "region_attributes",
            ]
        )

        # For each individual and time point
        for time_idx, time in enumerate(ds.time.values):
            for individual in ds.individuals.values:
                # Get position and shape data
                pos = ds.position.sel(time=time, individuals=individual).values
                shape = ds.shape.sel(time=time, individuals=individual).values

                # Skip if NaN values
                if np.isnan(pos).any() or np.isnan(shape).any():
                    continue

                # Write row
                writer.writerow(
                    _write_single_via_row(
                        time_idx, individual, pos, shape, video_id
                    )
                )

    logger.info(f"Saved bounding boxes dataset to {file.path}.")
    return file.path
