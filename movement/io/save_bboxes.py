"""Save bounding boxes data from ``movement`` to VIA tracks .csv format."""

import csv
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from movement.utils.logging import logger
from movement.validators.datasets import ValidBboxesInputs
from movement.validators.files import _validate_file_path

if TYPE_CHECKING:
    import _csv


def to_via_tracks_file(
    ds: xr.Dataset,
    file_path: str | Path,
    track_ids_from_trailing_numbers: bool = True,
    frame_n_digits: int | None = None,
    image_file_prefix: str | None = None,
    image_file_suffix: str = ".png",
) -> Path:
    """Save a ``movement`` bounding boxes dataset to a VIA tracks .csv file.

    Parameters
    ----------
    ds : xarray.Dataset
        The ``movement`` bounding boxes dataset to export.
    file_path : str or pathlib.Path
        Path where the VIA tracks .csv file [1]_ will be saved.
    track_ids_from_trailing_numbers : bool, optional
        If True, extract track IDs from the numbers at the end of the
        individuals' names (e.g. `mouse_1` -> track ID 1). If False, the
        track IDs will be assigned sequentially (0, 1, 2, ...) based on
        the alphabetically sorted list of individuals' names. Default is True.
    frame_n_digits : int, optional
        The number of digits to use to represent frame numbers in the image
        filenames (including leading zeros). If None, the number of digits is
        automatically determined from the largest frame number in the dataset,
        plus one (to have at least one leading zero). Default is None.
    image_file_prefix : str, optional
        Prefix to apply to every image filename. It is prepended to the frame
        number which is padded with leading zeros. If None or an empty string,
        nothing will be prepended to the padded frame number. Default is None.
    image_file_suffix : str, optional
        Suffix to add to every image filename holding the file extension.
        Strings with or without the dot are accepted. Default is '.png'.

    Returns
    -------
    pathlib.Path
        Path to the saved file.

    Notes
    -----
    The input arguments that define how the image filenames are formatted
    (``frame_n_digits``, ``image_file_prefix``, and ``image_file_suffix``)
    are useful to ensure the exported VIA tracks .csv file can be loaded in
    the VIA software alongside the image files the tracks refer to.

    References
    ----------
    .. [1] https://www.robots.ox.ac.uk/~vgg/software/via/docs/face_track_annotation.html

    Examples
    --------
    Export a ``movement`` bounding boxes dataset as a VIA tracks .csv file,
    deriving the track IDs from the numbers at the end of the individuals'
    names and assuming the image files are PNG files. The frame numbers in the
    image filenames are padded with at least one leading zero by default:

    >>> from movement.io import save_bboxes
    >>> save_bboxes.to_via_tracks_file(ds, "/path/to/output.csv")

    Export a ``movement`` bounding boxes dataset as a VIA tracks .csv file,
    assigning the track IDs sequentially based on the alphabetically sorted
    list of individuals' names, and assuming the image files are PNG files:

    >>> from movement.io import save_bboxes
    >>> save_bboxes.to_via_tracks_file(
    ...     ds,
    ...     "/path/to/output.csv",
    ...     track_ids_from_trailing_numbers=False,
    ... )

    Export a ``movement`` bounding boxes dataset as a VIA tracks .csv file,
    deriving the track IDs from the numbers at the end of the individuals'
    names, and assuming the image files are JPG files:

    >>> from movement.io import save_bboxes
    >>> save_bboxes.to_via_tracks_file(
    ...     ds,
    ...     "/path/to/output.csv",
    ...     image_file_suffix=".jpg",
    ... )

    Export a ``movement`` bounding boxes dataset as a VIA tracks .csv file,
    deriving the track IDs from the numbers at the end of the individuals'
    names and with image filenames following the format
    ``frame-<frame_number>.jpg``:

    >>> from movement.io import save_bboxes
    >>> save_bboxes.to_via_tracks_file(
    ...     ds,
    ...     "/path/to/output.csv",
    ...     image_file_prefix="frame-",
    ...     image_file_suffix=".jpg",
    ... )

    Export a ``movement`` bounding boxes dataset as a VIA tracks .csv file,
    deriving the track IDs from the numbers at the end of the individuals'
    names, and with frame numbers in the image filenames represented using 4
    digits (i.e., image filenames would be ``0000.png``, ``0001.png``, etc.):

    >>> from movement.io import save_bboxes
    >>> save_bboxes.to_via_tracks_file(
    ...     ds,
    ...     "/path/to/output.csv",
    ...     frame_n_digits=4,
    ... )

    """
    # Validate file path and dataset
    file = _validate_file_path(file_path, expected_suffix=[".csv"])
    ValidBboxesInputs.validate(ds)

    # Check the number of digits required to represent the frame numbers
    frame_n_digits = _check_frame_required_digits(
        ds=ds, frame_n_digits=frame_n_digits
    )

    # Define format string for image filenames
    img_filename_template = _get_image_filename_template(
        frame_n_digits=frame_n_digits,
        image_file_prefix=image_file_prefix,
        image_file_suffix=image_file_suffix,
    )

    # Map individuals' names to track IDs
    map_individual_to_track_id = _compute_individuals_to_track_ids_map(
        ds.coords["individual"].values,
        track_ids_from_trailing_numbers,
    )

    # Write file
    _write_via_tracks_csv(
        ds,
        file.path,
        map_individual_to_track_id,
        img_filename_template,
    )

    logger.info(f"Saved bounding boxes dataset to {file.path}.")
    return file.path


def _get_image_filename_template(
    frame_n_digits: int,
    image_file_prefix: str | None,
    image_file_suffix: str,
) -> str:
    """Compute a format string for the images' filenames.

    The filenames of the images in the VIA tracks .csv file are derived from
    the frame numbers. Optionally, a prefix can be added to the frame number.
    The suffix refers to the file extension of the image files.

    Parameters
    ----------
    frame_n_digits : int
        Number of digits used to represent the frame number, including any
        leading zeros.
    image_file_prefix : str | None
        Prefix for each image filename, prepended to the frame number. If
        None or an empty string, nothing will be prepended.
    image_file_suffix : str
        Suffix to add to each image filename to represent the file extension.

    Returns
    -------
    str
        Format string for the images' filenames.

    """
    # Add the dot to the file extension if required
    if not image_file_suffix.startswith("."):
        image_file_suffix = f".{image_file_suffix}"

    # Add the prefix if not None or not an empty string
    image_file_prefix_modified = (
        f"{image_file_prefix}" if image_file_prefix else ""
    )

    # Define filename format string
    return (
        f"{image_file_prefix_modified}"
        f"{{:0{frame_n_digits}d}}"
        f"{image_file_suffix}"
    )


def _check_frame_required_digits(
    ds: xr.Dataset,
    frame_n_digits: int | None,
) -> int:
    """Check the number of digits to represent the frame number is valid.

    Parameters
    ----------
    ds : xarray.Dataset
        A movement dataset.
    frame_n_digits : int | None
        The proposed number of digits to use to represent the frame numbers
        in the image filenames (including leading zeros). If None, the number
        of digits is inferred based on the largest frame number in the dataset.

    Returns
    -------
    int
        The number of digits to use to represent the frame numbers in the
        image filenames (including leading zeros).

    Raises
    ------
    ValueError
        If the proposed number of digits is not enough to represent all the
        frame numbers.

    """
    # Compute minimum number of digits required to represent the
    # largest frame number
    if ds.time_unit == "seconds":
        max_frame_number = max((ds.time.values * ds.fps).astype(int))
    else:
        max_frame_number = max(ds.time.values)
    min_required_digits = len(str(max_frame_number))

    # If requested number of digits is None, infer automatically
    if frame_n_digits is None:
        return min_required_digits + 1  # pad with at least one zero
    elif frame_n_digits < min_required_digits:
        raise logger.error(
            ValueError(
                "The requested number of digits cannot be used to represent "
                f"all the frame numbers. Got {frame_n_digits}, but the "
                f"maximum frame number has {min_required_digits} digits."
            )
        )
    else:
        return frame_n_digits


def _compute_individuals_to_track_ids_map(
    individuals: Iterable[str],
    track_ids_from_trailing_numbers: bool,
) -> dict[str, int]:
    """Compute the map from individuals' names to track IDs.

    Parameters
    ----------
    individuals : Iterable[str]
        List of individuals' names.
    track_ids_from_trailing_numbers : bool
        If True, extract track ID from the last consecutive digits in
        the individuals' names. If False, the track IDs will be assigned
        sequentially (0, 1, 2, ...) based on the alphabetically
        sorted list of individuals' names.

    Returns
    -------
    dict[str, int]
        A dictionary mapping individuals' names to track IDs.

    """
    if track_ids_from_trailing_numbers:
        # Extract track IDs from the trailing numbers in the individuals' names
        map_individual_to_track_id = _extract_track_ids_from_individuals_names(
            individuals
        )
    else:
        # Assign track IDs sequentially based on the alphabetically sorted
        # list of individuals' names
        map_individual_to_track_id = {
            individual: i for i, individual in enumerate(sorted(individuals))
        }

    return map_individual_to_track_id


def _extract_track_ids_from_individuals_names(
    individuals: Iterable[str],
) -> dict[str, int]:
    """Extract track IDs as the last digits in the individuals' names.

    Parameters
    ----------
    individuals : Iterable[str]
        List of individuals' names.

    Returns
    -------
    dict[str, int]
        A dictionary mapping individuals' names to track IDs.

    Raises
    ------
    ValueError
        If a track ID is not found by looking at the last consecutive digits
        in an individual's name, or if the extracted track IDs cannot be
        uniquely mapped to the individuals' names.

    """
    map_individual_to_track_id = {}

    for individual in individuals:
        # Match the last consecutive digits in the individual's name
        # even if they are not at the end of the string
        pattern = r"(\d+)(?=\D*$)"
        match = re.search(pattern, individual)
        if match:
            track_id = int(match.group(1))
            map_individual_to_track_id[individual] = track_id
        else:
            raise logger.error(
                ValueError(f"Could not extract track ID from {individual}.")
            )

    # Check that all individuals have a unique track ID
    if len(set(map_individual_to_track_id.values())) != len(set(individuals)):
        raise logger.error(
            ValueError(
                "Could not extract a unique track ID for all individuals. "
                f"Expected {len(set(individuals))} unique track IDs, "
                f"but got {len(set(map_individual_to_track_id.values()))}."
            )
        )

    return map_individual_to_track_id


def _write_via_tracks_csv(
    ds: xr.Dataset,
    file_path: str | Path,
    map_individual_to_track_id: dict,
    img_filename_template: str,
) -> None:
    """Write a VIA tracks .csv file.

    Parameters
    ----------
    ds : xarray.Dataset
        A movement bounding boxes dataset.
    file_path : str or pathlib.Path
        Path where the VIA tracks .csv file will be saved.
    map_individual_to_track_id : dict
        Dictionary mapping individuals' names to track IDs.
    img_filename_template : str
        Format string for the images' filenames.

    """
    # Define VIA tracks .csv header
    header = [
        "filename",
        "file_size",
        "file_attributes",
        "region_count",
        "region_id",
        "region_shape_attributes",
        "region_attributes",
    ]

    # Get time values in frames
    if ds.time_unit == "seconds":
        time_in_frames = (ds.time.values * ds.fps).astype(int)
    else:
        time_in_frames = ds.time.values

    # Locate bboxes with null position or shape
    null_position_or_shape = np.any(ds.position.isnull(), axis=1) | np.any(
        ds.shape.isnull(), axis=1
    )  # (time, individuals)

    with open(file_path, "w", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(header)

        # Loop through frames
        for time_idx, time in enumerate(ds.time.values):
            frame_number = time_in_frames[time_idx]

            # Compute region count for current frame
            region_count = int(np.sum(~null_position_or_shape[time_idx, :]))

            # Initialise region ID for current frame
            region_id = 0

            # Loop through individuals
            for indiv in ds.individual.values:
                # Get position and shape data
                xy_data = ds.position.sel(time=time, individual=indiv).values
                wh_data = ds.shape.sel(time=time, individual=indiv).values

                # If the position or shape data contain NaNs, do not write
                # this bounding box to file
                if np.isnan(xy_data).any() or np.isnan(wh_data).any():
                    continue

                # Get confidence score
                confidence = ds.confidence.sel(
                    time=time, individual=indiv
                ).values
                if np.isnan(confidence):
                    confidence = None  # pass as None if confidence is NaN

                # Get track IDs from individuals' names
                track_id = map_individual_to_track_id[indiv]

                # Write row
                _write_single_row(
                    csv_writer,
                    xy_data,
                    wh_data,
                    confidence,
                    track_id,
                    region_count,
                    region_id,
                    img_filename_template.format(frame_number),
                    image_size=None,
                )

                # Update region ID for this frame
                region_id += 1


def _write_single_row(
    writer: "_csv._writer",  # requires a string literal type annotation
    xy_values: np.ndarray,
    wh_values: np.ndarray,
    confidence: float | None,
    track_id: int,
    region_count: int,
    region_id: int,
    img_filename: str,
    image_size: int | None,
) -> tuple[str, int, str, int, int, str, str]:
    """Write a single row of a VIA tracks .csv file and return it as a tuple.

    Parameters
    ----------
    writer : csv.writer
        CSV writer object.
    xy_values : np.ndarray
        Array with the x, y coordinates of the bounding box centroid.
    wh_values : np.ndarray
        Array with the width and height of the bounding box.
    confidence : float | None
        Confidence score for the bounding box detection.
    track_id : int
        Integer identifying a single track of bounding boxes across frames.
    region_count : int
        Total number of bounding boxes in the current frame.
    region_id : int
        Integer that identifies the bounding boxes in a frame starting from 0.
        Note that it is the result of an enumeration, and it does not
        necessarily match the track ID.
    img_filename : str
        Filename of the image file corresponding to the current frame.
    image_size : int | None
        File size in bytes. If None, the file size is set to 0.

    Returns
    -------
    tuple[str, int, str, int, int, str, str]
        A tuple with the data formatted for a single row in a VIA-tracks
        .csv file.

    Notes
    -----
    The reference for the VIA tracks .csv file format is at
    https://www.robots.ox.ac.uk/~vgg/software/via/docs/face_track_annotation.html

    """
    # Calculate top-left coordinates of bounding box
    x_center, y_center = xy_values
    width, height = wh_values
    x_top_left = x_center - width / 2
    y_top_left = y_center - height / 2

    # Define file attributes (placeholder value)
    file_attributes = json.dumps({"shot": 0})

    # Define region shape attributes
    region_shape_attributes = json.dumps(
        {
            "name": "rect",
            "x": float(x_top_left),
            "y": float(y_top_left),
            "width": float(width),
            "height": float(height),
        }
    )

    # Define region attributes
    region_attributes_dict: dict[str, float | int] = {"track": int(track_id)}
    if confidence is not None:
        # convert to float to ensure it is json-serializable
        region_attributes_dict["confidence"] = float(confidence)
    region_attributes = json.dumps(region_attributes_dict)

    # Set image size
    image_size = int(image_size) if image_size is not None else 0

    # Define row data
    row = (
        img_filename,
        image_size,
        file_attributes,
        region_count,
        region_id,
        region_shape_attributes,
        region_attributes,
    )

    writer.writerow(row)

    return row
