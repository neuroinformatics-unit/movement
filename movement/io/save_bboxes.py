"""Save bounding boxes tracking data from ``movement`` to various file formats."""

import logging
from pathlib import Path

import pandas as pd
import xarray as xr

from movement.utils.logging import log_error
from movement.validators.datasets import ValidBboxesDataset
from movement.validators.files import ValidFile

logger = logging.getLogger(__name__)


def _ds_to_via_tracks_df(ds: xr.Dataset) -> pd.DataFrame:
    """Convert a ``movement`` dataset to a VIA-tracks DataFrame.

    Parameters
    ----------
    ds : xarray.Dataset
        ``movement`` dataset containing bounding box tracks, confidence scores,
        and associated metadata.

    Returns
    -------
    pandas.DataFrame
        DataFrame in VIA-tracks format.

    Notes
    -----
    The VIA-tracks format expects the following columns:
    - frame_filename: Name of the frame file
    - region_id: Unique identifier for each bounding box
    - region_shape_attributes: Dictionary containing x, y, width, height
    - region_attributes: Dictionary containing confidence score

    """
    # Get the number of frames and individuals
    n_frames = ds.sizes["time"]
    n_individuals = ds.sizes["individuals"]

    # Create frame filenames (assuming zero-padded frame numbers)
    frame_filenames = [f"frame_{i:06d}.jpg" for i in range(n_frames)]

    # Initialize lists to store data
    data = []

    # For each frame and individual, create a row in the DataFrame
    for frame_idx in range(n_frames):
        frame_filename = frame_filenames[frame_idx]

        for individual_idx in range(n_individuals):
            # Get position and shape data
            x = ds.position[frame_idx, 0, individual_idx].item()
            y = ds.position[frame_idx, 1, individual_idx].item()
            width = ds.shape[frame_idx, 0, individual_idx].item()
            height = ds.shape[frame_idx, 1, individual_idx].item()
            confidence = ds.confidence[frame_idx, individual_idx].item()

            # Create region shape attributes dictionary
            region_shape_attributes = {
                "name": "rect",
                "x": x - width / 2,  # Convert from center to top-left
                "y": y - height / 2,
                "width": width,
                "height": height,
            }

            # Create region attributes dictionary
            region_attributes = {"confidence": confidence}

            # Create row data
            row = {
                "frame_filename": frame_filename,
                "region_id": individual_idx,
                "region_shape_attributes": str(region_shape_attributes),
                "region_attributes": str(region_attributes),
            }
            data.append(row)

    # Create DataFrame
    df = pd.DataFrame(data)
    return df


def to_via_tracks_file(
    ds: xr.Dataset,
    file_path: str | Path,
) -> None:
    """Save a ``movement`` dataset to a VIA-tracks file.

    Parameters
    ----------
    ds : xarray.Dataset
        ``movement`` dataset containing bounding box tracks, confidence scores,
        and associated metadata.
    file_path : pathlib.Path or str
        Path to the file to save the bounding boxes to. File extension must be .csv.

    Notes
    -----
    VIA-tracks saves bounding box tracking outputs as .csv files. The format
    includes frame filenames, region IDs, shape attributes (x, y, width, height),
    and region attributes (confidence scores).

    Examples
    --------
    >>> from movement.io import save_bboxes, load_bboxes
    >>> ds = load_bboxes.from_numpy(
    ...     position_array=np.random.rand(100, 2, 2),
    ...     shape_array=np.ones((100, 2, 2)) * [40, 30],
    ...     confidence_array=np.ones((100, 2)) * 0.5,
    ...     individual_names=["id_0", "id_1"],
    ... )
    >>> save_bboxes.to_via_tracks_file(ds, "path/to/file.csv")

    """
    # Validate file path
    file = _validate_file_path(file_path, expected_suffix=[".csv"])

    # Validate dataset
    _validate_dataset(ds)

    # Convert dataset to VIA-tracks DataFrame
    df = _ds_to_via_tracks_df(ds)

    # Save DataFrame to CSV
    df.to_csv(file.path, index=False)
    logger.info(f"Saved bounding boxes dataset to {file.path}.")


def _validate_file_path(
    file_path: str | Path, expected_suffix: list[str]
) -> ValidFile:
    """Validate the file path and return a ValidFile object.

    Parameters
    ----------
    file_path : pathlib.Path or str
        Path to the file to save the data to.
    expected_suffix : list of str
        List of expected file extensions.

    Returns
    -------
    ValidFile
        Validated file path object.

    Raises
    ------
    ValueError
        If the file path is invalid or has an unexpected extension.

    """
    file = ValidFile(file_path)
    if file.suffix not in expected_suffix:
        raise log_error(
            ValueError,
            f"Expected file extension to be one of {expected_suffix}, "
            f"but got {file.suffix}.",
        )
    return file


def _validate_dataset(ds: xr.Dataset) -> None:
    """Validate that the dataset is a valid ``movement`` bounding boxes dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to validate.

    Raises
    ------
    ValueError
        If the dataset is not a valid ``movement`` bounding boxes dataset.

    """
    try:
        ValidBboxesDataset(
            position_array=ds.position.data,
            shape_array=ds.shape.data,
            confidence_array=ds.confidence.data,
            individual_names=ds.coords["individuals"].data.tolist(),
            frame_array=ds.coords["time"].data,
            fps=ds.attrs.get("fps"),
            source_software=ds.attrs.get("source_software"),
        )
    except ValueError as e:
        raise log_error(ValueError, str(e))
