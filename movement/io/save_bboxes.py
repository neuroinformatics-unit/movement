"""Save bounding boxes data from ``movement`` to VIA-tracks CSV format."""

import csv
import re
from pathlib import Path

import numpy as np
import xarray as xr

from movement.io.save_poses import _validate_file_path
from movement.utils.logging import logger
from movement.validators.datasets import ValidBboxesDataset


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


def _map_individuals_to_track_ids(
    list_individuals: list[str], extract_track_id_from_individuals: bool
) -> dict[str, int]:
    """Map individuals to track IDs.

    Parameters
    ----------
    list_individuals : list[str]
        List of individuals.
    extract_track_id_from_individuals : bool
        If True, extract track_id from individuals' names. If False, the
        track_id will be factorised from the sorted individuals' names.

    Returns
    -------
    dict[str, int]
        A dictionary mapping individuals (str) to track IDs (int).

    """
    # Use sorted list of individuals' names
    list_individuals = sorted(list_individuals)

    # Map individuals to track IDs
    map_individual_to_track_id = {}
    if extract_track_id_from_individuals:
        # Extract consecutive integers at the end of individuals' names
        for individual in list_individuals:
            match = re.search(r"\d+$", individual)
            if match:
                map_individual_to_track_id[individual] = int(match.group())
            else:
                raise ValueError(
                    f"Could not extract track ID from {individual}."
                )

        # Check that all individuals have a track ID
        if len(set(map_individual_to_track_id.values())) != len(
            set(list_individuals)
        ):
            raise ValueError(
                "Could not extract a unique track ID for all individuals. "
                f"Expected {len(set(list_individuals))} unique track IDs, "
                f"but got {len(set(map_individual_to_track_id.values()))}."
            )

    else:
        # Factorise track IDs from sorted individuals' names
        map_individual_to_track_id = {
            individual: i for i, individual in enumerate(list_individuals)
        }

    return map_individual_to_track_id


def to_via_tracks_file(
    ds: xr.Dataset,
    file_path: str | Path,
    extract_track_id_from_individuals: bool = False,
    filename_prefix: str | None = None,
) -> Path:
    """Save a movement bounding boxes dataset to a VIA-tracks CSV file.

    Parameters
    ----------
    ds : xarray.Dataset
        The movement bounding boxes dataset to export.
    file_path : str or pathlib.Path
        Path where the VIA-tracks CSV file will be saved.
    extract_track_id_from_individuals : bool, optional
        If True, extract track_id from individuals' names. If False, the
        track_id will be factorised from the sorted individuals' names.
        Default is False.
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
    _validate_bboxes_dataset(ds)

    # Calculate the maximum number of digits required
    # to represent the frame number
    # (add 1 to prepend at least one zero)
    max_digits = len(str(ds.time.size)) + 1

    # Map individuals to track IDs
    individual_to_track_id = _map_individuals_to_track_ids(
        ds.coords["individuals"].values,
        extract_track_id_from_individuals,
    )

    # Write csv file
    with open(file.path, "w", newline="") as f:
        # Write header
        writer = csv.writer(f)
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
                # TODO: has to be an integer
                track_id = individual_to_track_id[individual]

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


def _validate_bboxes_dataset(ds: xr.Dataset) -> None:
    """Validate the input as a proper ``movement`` pose dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to validate.

    Raises
    ------
    TypeError
        If the input is not an xarray Dataset.
    ValueError
        If the dataset is missing required data variables or dimensions
        for a valid ``movement`` pose dataset.

    """
    if not isinstance(ds, xr.Dataset):
        raise logger.error(
            TypeError(f"Expected an xarray Dataset, but got {type(ds)}.")
        )

    missing_vars = set(ValidBboxesDataset.VAR_NAMES) - set(ds.data_vars)
    if missing_vars:
        raise ValueError(
            f"Missing required data variables: {sorted(missing_vars)}"
        )  # sort for a reproducible error message

    missing_dims = set(ValidBboxesDataset.DIM_NAMES) - set(ds.dims)
    if missing_dims:
        raise ValueError(
            f"Missing required dimensions: {sorted(missing_dims)}"
        )  # sort for a reproducible error message
