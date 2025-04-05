"""Save bounding boxes data from ``movement`` to VIA-tracks CSV format."""

import csv
import json
from pathlib import Path

import numpy as np
import xarray as xr

from movement.utils.logging import logger
from movement.validators.datasets import ValidBboxesDataset
from movement.validators.files import ValidFile


def _validate_dataset(ds: xr.Dataset) -> None:
    """Validate the input as a proper ``movement`` bounding boxes dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to validate.

    Raises
    ------
    TypeError
        If the input is not an xarray Dataset.
    ValueError
        If the dataset is missing required data variables or dimensions.

    """
    if not isinstance(ds, xr.Dataset):
        error_msg = f"Expected an xarray Dataset, but got {type(ds)}."
        raise logger.error(TypeError(error_msg))

    missing_vars = set(ValidBboxesDataset.VAR_NAMES) - set(ds.data_vars)
    if missing_vars:
        error_msg = f"Missing required data variables: {sorted(missing_vars)}"
        raise logger.error(ValueError(error_msg))

    missing_dims = set(ValidBboxesDataset.DIM_NAMES) - set(ds.dims)
    if missing_dims:
        error_msg = f"Missing required dimensions: {sorted(missing_dims)}"
        raise logger.error(ValueError(error_msg))


def _validate_file_path(
    file_path: str | Path, expected_suffix: list[str]
) -> ValidFile:
    """Validate the input file path.

    Parameters
    ----------
    file_path : pathlib.Path or str
        Path to the file to validate.
    expected_suffix : list of str
        Expected suffix(es) for the file.

    Returns
    -------
    ValidFile
        The validated file.

    Raises
    ------
    OSError
        If the file cannot be written.
    ValueError
        If the file does not have the expected suffix.

    """
    try:
        file = ValidFile(
            file_path,
            expected_permission="w",
            expected_suffix=expected_suffix,
        )
    except (OSError, ValueError) as error:
        raise logger.error(error) from error
    return file


def _prepare_via_row(
    frame: int,
    individual: str,
    pos: np.ndarray,
    shape: np.ndarray,
    video_id: str,
) -> list:
    """Prepare a single row for the VIA-tracks CSV file.

    Parameters
    ----------
    frame : int
        Frame number.
    individual : str
        Individual identifier.
    pos : np.ndarray
        Position data (x, y).
    shape : np.ndarray
        Shape data (width, height).
    video_id : str
        Video identifier.

    Returns
    -------
    list
        Row data in VIA-tracks format.

    """
    # Calculate top-left coordinates
    x_center, y_center = pos
    width, height = shape
    x = x_center - width / 2
    y = y_center - height / 2

    # Prepare region shape attributes
    region_shape_attributes = json.dumps(
        {
            "name": "rect",
            "x": int(x),
            "y": int(y),
            "width": int(width),
            "height": int(height),
        }
    )

    # Prepare region attributes
    region_attributes = json.dumps({"track": individual})

    return [
        f"{video_id}_{frame:06d}.jpg",  # filename
        0,  # file_size (placeholder)
        "{}",  # file_attributes (empty JSON object)
        1,  # region_count
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
        Video identifier to use in the export. If None, will use the filename.

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
    file = _validate_file_path(file_path, expected_suffix=[".csv"])
    _validate_dataset(ds)

    # Use filename as video_id if not provided
    if video_id is None:
        video_id = file.path.stem

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
        for frame, time in enumerate(ds.time.values):
            for individual in ds.individuals.values:
                # Get position and shape data
                pos = ds.position.sel(time=time, individuals=individual).values
                shape = ds.shape.sel(time=time, individuals=individual).values

                # Skip if NaN values
                if np.isnan(pos).any() or np.isnan(shape).any():
                    continue

                # Write row
                writer.writerow(
                    _prepare_via_row(frame, individual, pos, shape, video_id)
                )

    logger.info(f"Saved bounding boxes dataset to {file.path}.")
    return file.path
