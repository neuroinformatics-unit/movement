"""Save bounding boxes data from ``movement`` to VIA-tracks CSV format."""

import _csv
import csv
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
        If True, extract track_id from individuals' names. If False, the
        track_id will be factorised from the sorted individuals' names.
        Default is False.
    image_file_prefix : str, optional
        Prefix for each image filename, prepended to frame number. If None or
        an empty string, nothing will be prepended.
    image_file_suffix : str, optional
        Suffix to add to each image filename. Default is '.png'.

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

    # Define format string for image filenames
    img_filename_template = _get_image_filename_template(
        frame_max_digits=int(np.ceil(np.log10(ds.time.size)) + 1),
        image_file_prefix=image_file_prefix,
        image_file_suffix=image_file_suffix,
    )

    # Map individuals' names to track IDs
    map_individual_to_track_id = _get_map_individuals_to_track_ids(
        ds.coords["individuals"].values,
        extract_track_id_from_individuals,
    )

    # Write csv file
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
    """Compute a format string for the image filename.

    Parameters
    ----------
    frame_max_digits : int
        Maximum number of digits in the frame number.
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
    # Add dot to image_file_suffix if required
    if not image_file_suffix.startswith("."):
        image_file_suffix = f".{image_file_suffix}"

    # Add prefix to image_file_prefix if required
    image_file_prefix_modified = (
        f"{image_file_prefix}" if image_file_prefix else ""
    )

    # Define filename format string
    return (
        f"{image_file_prefix_modified}"
        f"{{:0{frame_max_digits}d}}"  #
        f"{image_file_suffix}"
    )


def _get_map_individuals_to_track_ids(
    list_individuals: list[str], extract_track_id_from_individuals: bool
) -> dict[str, int]:
    """Map individuals' names to track IDs.

    Parameters
    ----------
    list_individuals : list[str]
        List of individuals' names.
    extract_track_id_from_individuals : bool
        If True, extract track ID from individuals' names. If False, the
        track ID will be factorised from the sorted list of individuals' names.

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
        # Look for consecutive integers at the end of the individuals' names
        for individual in list_individuals:
            # Find the first non-digit character starting from the end
            last_idx = len(individual) - 1
            first_non_digit_idx = last_idx
            while (
                first_non_digit_idx >= 0
                and individual[first_non_digit_idx].isdigit()
            ):
                first_non_digit_idx -= 1

            # Extract track ID from first digit character until the end
            if first_non_digit_idx < last_idx:
                track_id = int(individual[first_non_digit_idx + 1 :])
                map_individual_to_track_id[individual] = track_id
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

                # Skip if there are NaN values
                if np.isnan(xy_data).any() or np.isnan(wh_data).any():
                    continue

                # Get confidence score
                confidence = ds.confidence.sel(
                    time=time, individuals=indiv
                ).values

                # Get track IDs from individuals' names
                track_id = map_individual_to_track_id[indiv]

                # Write row
                _write_single_row(
                    csv_writer,
                    xy_data,
                    wh_data,
                    confidence if not np.isnan(confidence) else None,
                    track_id,
                    time_in_frames[time_idx],
                    img_filename_template,
                    image_size=None,
                )


def _write_single_row(
    writer: "_csv._writer",  # requires a string literal type annotation
    xy_coordinates: np.ndarray,
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
    xy_coordinates : np.ndarray
        Bounding box centroid position data (x, y).
    wh_values : np.ndarray
        Bounding box shape data (width, height).
    confidence : float | None
        Confidence score.
    track_id : int
        Integer identifying a single track.
    frame_number : int
        Frame number.
    img_filename_template : str
        Format string for each image filename.
    image_size : int | None
        File size in bytes. If None, the file size is set to 0.

    Returns
    -------
    tuple[str, int, str, int, int, str, str]
        Data formatted for a single row in a VIA-tracks .csv file.

    """
    # Calculate top-left coordinates of bounding box
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

    # Define row data
    row = (
        img_filename_template.format(frame_number),  # filename
        image_size if image_size is not None else 0,  # file size
        "{}",  # file_attributes placeholder
        0,  # region_count placeholder
        0,  # region_id placeholder
        f"{region_shape_attributes}",
        f"{region_attributes}",
    )

    writer.writerow(row)

    return row
