"""Save bounding boxes data from ``movement`` to VIA-tracks CSV format."""

import csv
from pathlib import Path

import numpy as np
import xarray as xr

from movement.io.save_poses import _validate_dataset, _validate_file_path
from movement.utils.logging import logger


def _write_single_via_row(
    frame_idx: int,
    track_id: int,
    xy_coordinates: np.ndarray,
    wh_values: np.ndarray,
    max_digits: int,
    confidence: float | None = None,
    filename_prefix: str | None = None,
    all_frames_size: int | None = None,
) -> tuple[str, int, str, int, int, str, str]:
    """Return a list representing a single row for the VIA-tracks CSV file.

    Parameters
    ----------
    frame_idx : int
        Frame index (0-based).
    track_id : int
        Integer identifying a single track.
    xy_coordinates : np.ndarray
        Position data (x, y).
    wh_values : np.ndarray
        Shape data (width, height).
    max_digits : int
        Maximum number of digits in the frame number. Used to pad the frame
        number with zeros.
    confidence: float | None, optional
        Confidence score. Default is None.
    filename_prefix : str | None, optional
        Prefix for the filename, prepended to frame number. If None, nothing
        is prepended to the frame number.
    all_frames_size : int, optional
        Size (in bytes) of all frames in the video. Default is 0.

    Returns
    -------
    tuple[str, int, str, int, int, str, str]
        Data formatted for a single row in a VIA-tracks .csv file.

    """
    # Calculate top-left coordinates
    x_center, y_center = xy_coordinates
    width, height = wh_values
    x_top_left = x_center - width / 2
    y_top_left = y_center - height / 2

    # Define region shape attributes
    region_shape_attributes = {
        "name": "rect",
        "x": float(x_top_left),
        "y": float(y_top_left),
        "width": float(width),
        "height": float(height),
    }

    # Define region attributes
    if confidence is not None:
        region_attributes = (
            f'{{"track":"{int(track_id)}", "confidence":"{confidence}"}}'
        )
    else:
        region_attributes = f'{{"track":"{int(track_id)}"}}'

    # Define filename
    filename_prefix = f"{filename_prefix}_" if filename_prefix else ""
    filename = f"{filename_prefix}{frame_idx:0{max_digits}d}.jpg"

    # Define row data
    return (
        filename,  # filename
        all_frames_size if all_frames_size is not None else 0,  # frame size
        "{}",  # file_attributes ---can this be empty?
        # if not: '{{"clip":{}}}'.format("000"),
        0,  # region_count -- should this be 0?
        0,  # region_id
        f"{region_shape_attributes}",  # region_shape_attributes
        f"{region_attributes}",  # region_attributes
    )


def to_via_tracks_file(
    ds: xr.Dataset,
    file_path: str | Path,
    filename_prefix: str | None = None,
) -> Path:
    """Save a movement bounding boxes dataset to a VIA-tracks CSV file.

    Parameters
    ----------
    ds : xarray.Dataset
        The movement bounding boxes dataset to export.
    file_path : str or pathlib.Path
        Path where the VIA-tracks CSV file will be saved.
    filename_prefix : str, optional
        Prefix for each image filename, prepended to frame number. If None,
        nothing will be prepended.

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

    # Calculate the maximum number of digits required
    # to represent the frame number
    # (add 1 to prepend at least one zero)
    max_digits = len(str(ds.time.size)) + 1

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
                xy = ds.position.sel(time=time, individuals=individual).values
                wh = ds.shape.sel(time=time, individuals=individual).values

                # Get confidence score
                confidence = ds.confidence.sel(
                    time=time, individuals=individual
                ).values

                # Get track_id from individual
                track_id = ds.tracks.sel(individuals=individual).values

                # Skip if NaN values
                if np.isnan(xy).any() or np.isnan(wh).any():
                    continue

                # # Write row
                writer.writerow(
                    _write_single_via_row(
                        time_idx,
                        track_id,
                        xy,
                        wh,
                        max_digits,
                        confidence if np.isnan(confidence) else None,
                        filename_prefix,
                    )
                )

    logger.info(f"Saved bounding boxes dataset to {file.path}.")
    return file.path
