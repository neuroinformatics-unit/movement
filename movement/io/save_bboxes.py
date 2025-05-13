"""Save bounding boxes data from ``movement`` to VIA-tracks CSV format."""

import _csv
import csv
import json
from pathlib import Path

import numpy as np
import xarray as xr

from movement.io.utils import _validate_file_path
from movement.utils.logging import logger
from movement.validators.datasets import ValidBboxesDataset


def to_via_tracks_file(
    ds: xr.Dataset,
    file_path: str | Path,
    extract_track_id_from_individuals: bool = False,
    image_file_prefix: str | None = None,
    image_file_suffix: str = ".png",
) -> Path:
    """Save a movement bounding boxes dataset to a VIA-tracks CSV file.

    Parameters
    ----------
    ds : xarray.Dataset
        The movement bounding boxes dataset to export.
    file_path : str or pathlib.Path
        Path where the VIA-tracks CSV file will be saved.
    extract_track_id_from_individuals : bool, optional
        If True, extract track IDs from the numbers at the end of the
        individuals' names (e.g. `mouse_1` -> track ID 1). If False, the
        track IDs will be factorised from the list of sorted individuals'
        names. Default is False.
    image_file_prefix : str, optional
        Prefix to apply to every image filename. It is prepended to the frame
        number which is padded with leading zeros. If None or an empty string,
        nothing will be prepended to the padded frame number. Default is None.
    image_file_suffix : str, optional
        Suffix to add to each image filename holding the file extension.
        Strings with or without the dot are accepted. Default is '.png'.

    Returns
    -------
    pathlib.Path
        Path to the saved file.

    Examples
    --------
    Export a ``movement`` bounding boxes dataset as a VIA-tracks CSV file,
    deriving the track IDs from the list of sorted individuals and assuming
    the image files are PNG files:
    >>> from movement.io import save_boxes
    >>> save_boxes.to_via_tracks_file(ds, "/path/to/output.csv")

    Export a ``movement`` bounding boxes dataset as a VIA-tracks CSV file,
    extracting track IDs from the end of the individuals' names and assuming
    the image files are JPG files:
    >>> save_boxes.to_via_tracks_file(
    ...     ds,
    ...     "/path/to/output.csv",
    ...     extract_track_id_from_individuals=True,
    ...     image_file_suffix=".jpg",
    ... )

    Export a ``movement`` bounding boxes dataset as a VIA-tracks CSV file,
    with image filenames following the format ``frame_{frame_number}.jpg``
    and the track IDs derived from the list of sorted individuals:
    >>> save_boxes.to_via_tracks_file(
    ...     ds,
    ...     "/path/to/output.csv",
    ...     image_file_prefix="frame_",
    ...     image_file_suffix=".jpg",
    ... )

    """
    # Validate file path and dataset
    file = _validate_file_path(file_path, expected_suffix=[".csv"])
    _validate_bboxes_dataset(ds)

    # Define format string for image filenames
    img_filename_template = _get_image_filename_template(
        frame_max_digits=int(np.ceil(np.log10(ds.time.size))),
        image_file_prefix=image_file_prefix,
        image_file_suffix=image_file_suffix,
    )

    # Map individuals' names to track IDs
    map_individual_to_track_id = _get_map_individuals_to_track_ids(
        ds.coords["individuals"].values,
        extract_track_id_from_individuals,
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


def _get_image_filename_template(
    frame_max_digits: int,
    image_file_prefix: str | None,
    image_file_suffix: str,
) -> str:
    """Compute a format string for the images' filenames.

    The filenames of the images in the VIA-tracks CSV file are computed from
    the frame number which is padded with at least one leading zero.
    Optionally, a prefix can be added to the padded frame number. The suffix
    refers to the file extension of the image files.

    Parameters
    ----------
    frame_max_digits : int
        Maximum number of digits used to represent the frame number.
    image_file_prefix : str | None
        Prefix for each image filename, prepended to frame number. If None or
        an empty string, nothing will be prepended.
    image_file_suffix : str
        Suffix to add to each image filename.

    Returns
    -------
    str
        Format string for each image filename.

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
        f"{{:0{frame_max_digits + 1}d}}"  # +1 to pad with at least one zero
        f"{image_file_suffix}"
    )


def _get_map_individuals_to_track_ids(
    list_individuals: list[str],
    extract_track_id_from_individuals: bool,
) -> dict[str, int]:
    """Map individuals' names to track IDs.

    Parameters
    ----------
    list_individuals : list[str]
        List of individuals' names.
    extract_track_id_from_individuals : bool
        If True, extract track ID from the last consecutive digits in
        the individuals' names. If False, the track IDs will be factorised
        from the sorted list of individuals' names.

    Returns
    -------
    dict[str, int]
        A dictionary mapping individuals' names (str) to track IDs (int).

    Raises
    ------
    ValueError
        If extract_track_id_from_individuals is True and:
        - a track ID is not found by looking at the last consecutive digits
          in an individual's name, or
        - the extracted track IDs cannot be uniquely mapped to the
          individuals' names.

    """
    if extract_track_id_from_individuals:
        # Extract track IDs from the individuals' names
        map_individual_to_track_id = _get_track_id_from_individuals(
            list_individuals
        )
    else:
        # Factorise track IDs from sorted individuals' names
        list_individuals = sorted(list_individuals)
        map_individual_to_track_id = {
            individual: i for i, individual in enumerate(list_individuals)
        }

    return map_individual_to_track_id


def _get_track_id_from_individuals(
    list_individuals: list[str],
) -> dict[str, int]:
    """Extract track IDs as the last digits in the individuals' names.

    Parameters
    ----------
    list_individuals : list[str]
        List of individuals' names.

    Returns
    -------
    dict[str, int]
        A dictionary mapping individuals' names (str) to track IDs (int).

    Raises
    ------
    ValueError
        If a track ID is not found by looking at the last consecutive digits
        in an individual's name, or if the extracted track IDs cannot be
        uniquely mapped to the individuals' names.

    """
    map_individual_to_track_id = {}

    for individual in list_individuals:
        # Find the first non-digit character starting from the end
        last_idx = len(individual) - 1
        first_non_digit_idx = last_idx
        while (
            first_non_digit_idx >= 0
            and individual[first_non_digit_idx].isdigit()
        ):
            first_non_digit_idx -= 1

        # Extract track ID from (first_non_digit_idx+1) until the end
        if first_non_digit_idx < last_idx:
            track_id = int(individual[first_non_digit_idx + 1 :])
            map_individual_to_track_id[individual] = track_id
        else:
            raise ValueError(f"Could not extract track ID from {individual}.")

    # Check that all individuals have a unique track ID
    if len(set(map_individual_to_track_id.values())) != len(
        set(list_individuals)
    ):
        raise ValueError(
            "Could not extract a unique track ID for all individuals. "
            f"Expected {len(set(list_individuals))} unique track IDs, "
            f"but got {len(set(map_individual_to_track_id.values()))}."
        )

    return map_individual_to_track_id


def _write_via_tracks_csv(
    ds: xr.Dataset,
    file_path: str | Path,
    map_individual_to_track_id: dict,
    img_filename_template: str,
) -> None:
    """Write a VIA-tracks CSV file.

    Parameters
    ----------
    ds : xarray.Dataset
        The movement bounding boxes dataset to export.
    file_path : str or pathlib.Path
        Path where the VIA-tracks CSV file will be saved.
    map_individual_to_track_id : dict
        Dictionary mapping individual names to track IDs.
    img_filename_template : str
        Format string for each image filename.

    """
    # Define VIA-tracks CSV header
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

    with open(file_path, "w", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(header)

        # Write bbox data for each time point and individual
        for time_idx, time in enumerate(ds.time.values):
            for indiv in ds.individuals.values:
                # Get position and shape data
                xy_data = ds.position.sel(time=time, individuals=indiv).values
                wh_data = ds.shape.sel(time=time, individuals=indiv).values

                # If the position or shape data contains NaNs, do not write
                # this annotation
                if np.isnan(xy_data).any() or np.isnan(wh_data).any():
                    continue

                # Get confidence score
                confidence = ds.confidence.sel(
                    time=time, individuals=indiv
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
                    time_in_frames[time_idx],
                    img_filename_template,
                    image_size=None,
                )


def _write_single_row(
    writer: "_csv._writer",  # requires a string literal type annotation
    xy_values: np.ndarray,
    wh_values: np.ndarray,
    confidence: float | None,
    track_id: int,
    frame_number: int,
    img_filename_template: str,
    image_size: int | None,
) -> tuple[str, int, str, int, int, str, str]:
    """Return a tuple representing a single row of a VIA-tracks CSV file.

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
    frame_number : int
        Frame number.
    img_filename_template : str
        Format string to apply to the image filename. The image filename is
        formatted as the frame number padded with at least one leading zero,
        plus the file extension. Optionally, a prefix can be added to the
        padded frame number.
    image_size : int | None
        File size in bytes. If None, the file size is set to 0.

    Returns
    -------
    tuple[str, int, str, int, int, str, str]
        A tuple with the data formatted for a single row in a VIA-tracks
        .csv file.

    Notes
    -----
    The reference for the VIA-tracks CSV file format is taken from
    https://www.robots.ox.ac.uk/~vgg/software/via/docs/face_track_annotation.html

    """
    # Calculate top-left coordinates of bounding box
    x_center, y_center = xy_values
    width, height = wh_values
    x_top_left = x_center - width / 2
    y_top_left = y_center - height / 2

    # Define file attributes (placeholder value)
    # file_attributes = f'{{"shot": {0}}}'
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
        region_attributes_dict["confidence"] = float(confidence)
        # convert to float to ensure json-serializable
    region_attributes = json.dumps(region_attributes_dict)

    # Set image size
    image_size = int(image_size) if image_size is not None else 0

    # Define row data
    row = (
        img_filename_template.format(frame_number),
        image_size,
        file_attributes,
        0,  # region_count placeholder
        0,  # region_id placeholder
        region_shape_attributes,
        region_attributes,
    )

    writer.writerow(row)

    return row
