import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal, Optional, Union

import h5py
import numpy as np
from attrs import converters, define, field, validators

from movement.logging import log_error, log_warning


@define
class ValidFile:
    """Class for validating file paths.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the file.
    expected_permission : {'r', 'w', 'rw'}
        Expected access permission(s) for the file. If 'r', the file is
        expected to be readable. If 'w', the file is expected to be writable.
        If 'rw', the file is expected to be both readable and writable.
        Default: 'r'.
    expected_suffix : list of str
        Expected suffix(es) for the file. If an empty list (default), this
        check is skipped.

    Raises
    ------
    IsADirectoryError
        If the path points to a directory.
    PermissionError
        If the file does not have the expected access permission(s).
    FileNotFoundError
        If the file does not exist when `expected_permission` is 'r' or 'rw'.
    FileExistsError
        If the file exists when `expected_permission` is 'w'.
    ValueError
        If the file does not have one of the expected suffix(es).
    """

    path: Path = field(converter=Path, validator=validators.instance_of(Path))
    expected_permission: Literal["r", "w", "rw"] = field(
        default="r", validator=validators.in_(["r", "w", "rw"]), kw_only=True
    )
    expected_suffix: list[str] = field(factory=list, kw_only=True)

    @path.validator
    def path_is_not_dir(self, attribute, value):
        """Ensures that the path does not point to a directory."""
        if value.is_dir():
            raise log_error(
                IsADirectoryError,
                f"Expected a file path but got a directory: {value}.",
            )

    @path.validator
    def file_exists_when_expected(self, attribute, value):
        """Ensures that the file exists (or not) depending on the expected
        usage (read and/or write)."""
        if "r" in self.expected_permission:
            if not value.exists():
                raise log_error(
                    FileNotFoundError, f"File {value} does not exist."
                )
        else:  # expected_permission is 'w'
            if value.exists():
                raise log_error(
                    FileExistsError, f"File {value} already exists."
                )

    @path.validator
    def file_has_access_permissions(self, attribute, value):
        """Ensures that the file has the expected access permission(s).
        Raises a PermissionError if not."""
        file_is_readable = os.access(value, os.R_OK)
        parent_is_writeable = os.access(value.parent, os.W_OK)
        if ("r" in self.expected_permission) and (not file_is_readable):
            raise log_error(
                PermissionError,
                f"Unable to read file: {value}. "
                "Make sure that you have read permissions.",
            )
        if ("w" in self.expected_permission) and (not parent_is_writeable):
            raise log_error(
                PermissionError,
                f"Unable to write to file: {value}. "
                "Make sure that you have write permissions.",
            )

    @path.validator
    def file_has_expected_suffix(self, attribute, value):
        """Ensures that the file has one of the expected suffix(es)."""
        if self.expected_suffix:  # list is not empty
            if value.suffix not in self.expected_suffix:
                raise log_error(
                    ValueError,
                    f"Expected file with suffix(es) {self.expected_suffix} "
                    f"but got suffix {value.suffix} instead.",
                )


@define
class ValidHDF5:
    """Class for validating HDF5 files.

    Parameters
    ----------
    path : pathlib.Path
        Path to the HDF5 file.
    expected_datasets : list of str or None
        List of names of the expected datasets in the HDF5 file. If an empty
        list (default), this check is skipped.

    Raises
    ------
    ValueError
        If the file is not in HDF5 format or if it does not contain the
        expected datasets.
    """

    path: Path = field(validator=validators.instance_of(Path))
    expected_datasets: list[str] = field(factory=list, kw_only=True)

    @path.validator
    def file_is_h5(self, attribute, value):
        """Ensure that the file is indeed in HDF5 format."""
        try:
            with h5py.File(value, "r") as f:
                f.close()
        except Exception as e:
            raise log_error(
                ValueError,
                f"File {value} does not seem to be in valid" "HDF5 format.",
            ) from e

    @path.validator
    def file_contains_expected_datasets(self, attribute, value):
        """Ensure that the HDF5 file contains the expected datasets."""
        if self.expected_datasets:
            with h5py.File(value, "r") as f:
                diff = set(self.expected_datasets).difference(set(f.keys()))
                if len(diff) > 0:
                    raise log_error(
                        ValueError,
                        f"Could not find the expected dataset(s) {diff} "
                        f"in file: {value}. ",
                    )


@define
class ValidPosesCSV:
    """Class for validating CSV files that contain pose estimation outputs.
    in DeepLabCut format.

    Parameters
    ----------
    path : pathlib.Path
        Path to the CSV file.

    Raises
    ------
    ValueError
        If the CSV file does not contain the expected DeepLabCut index column
        levels among its top rows.
    """

    path: Path = field(validator=validators.instance_of(Path))

    @path.validator
    def csv_file_contains_expected_levels(self, attribute, value):
        """Ensure that the CSV file contains the expected index column levels
        among its top rows."""
        expected_levels = ["scorer", "bodyparts", "coords"]

        with open(value, "r") as f:
            top4_row_starts = [f.readline().split(",")[0] for _ in range(4)]

            if top4_row_starts[3].isdigit():
                # if 4th row starts with a digit, assume single-animal DLC file
                expected_levels.append(top4_row_starts[3])
            else:
                # otherwise, assume multi-animal DLC file
                expected_levels.insert(1, "individuals")

            if top4_row_starts != expected_levels:
                raise log_error(
                    ValueError,
                    "CSV header rows do not match the known format for "
                    "DeepLabCut pose estimation output files.",
                )


def _list_of_str(value: Union[str, Iterable[Any]]) -> list[str]:
    """Try to coerce the value into a list of strings.
    Otherwise, raise a ValueError."""
    if isinstance(value, str):
        log_warning(
            f"Invalid value ({value}). Expected a list of strings. "
            "Converting to a list of length 1."
        )
        return [value]
    elif isinstance(value, Iterable):
        return [str(item) for item in value]
    else:
        raise log_error(
            ValueError, f"Invalid value ({value}). Expected a list of strings."
        )


def _ensure_type_ndarray(value: Any) -> None:
    """Raise ValueError the value is a not numpy array."""
    if not isinstance(value, np.ndarray):
        raise log_error(
            ValueError, f"Expected a numpy array, but got {type(value)}."
        )


def _set_fps_to_none_if_invalid(fps: Optional[float]) -> Optional[float]:
    """Set fps to None if a non-positive float is passed."""
    if fps is not None and fps <= 0:
        log_warning(
            f"Invalid fps value ({fps}). Expected a positive number. "
            "Setting fps to None."
        )
        return None
    return fps


def _validate_list_length(
    attribute: str, value: Optional[list], expected_length: int
):
    """Raise a ValueError if the list does not have the expected length."""
    if (value is not None) and (len(value) != expected_length):
        raise log_error(
            ValueError,
            f"Expected `{attribute}` to have length {expected_length}, "
            f"but got {len(value)}.",
        )


@define(kw_only=True)
class ValidPoseTracks:
    """Class for validating pose tracking data imported from a file.

    Attributes
    ----------
    tracks_array : np.ndarray
        Array of shape (n_frames, n_individuals, n_keypoints, n_space)
        containing the pose tracks. It will be converted to a
        `xarray.DataArray` object named "pose_tracks".
    scores_array : np.ndarray, optional
        Array of shape (n_frames, n_individuals, n_keypoints) containing
        the point-wise confidence scores. It will be converted to a
        `xarray.DataArray` object named "confidence".
        If None (default), the scores will be set to an array of NaNs.
    individual_names : list of str, optional
        List of unique names for the individuals in the video. If None
        (default), the individuals will be named "individual_0",
        "individual_1", etc.
    keypoint_names : list of str, optional
        List of unique names for the keypoints in the skeleton. If None
        (default), the keypoints will be named "keypoint_0", "keypoint_1",
        etc.
    fps : float, optional
        Frames per second of the video. Defaults to None.
    source_software : str, optional
        Name of the software from which the pose tracks were loaded.
        Defaults to None.
    """

    # Define class attributes
    tracks_array: np.ndarray = field()
    scores_array: Optional[np.ndarray] = field(default=None)
    individual_names: Optional[list[str]] = field(
        default=None,
        converter=converters.optional(_list_of_str),
    )
    keypoint_names: Optional[list[str]] = field(
        default=None,
        converter=converters.optional(_list_of_str),
    )
    fps: Optional[float] = field(
        default=None,
        converter=converters.pipe(  # type: ignore
            converters.optional(float), _set_fps_to_none_if_invalid
        ),
    )
    source_software: Optional[str] = field(
        default=None,
        validator=validators.optional(validators.instance_of(str)),
    )

    # Add validators
    @tracks_array.validator
    def _validate_tracks_array(self, attribute, value):
        _ensure_type_ndarray(value)
        if value.ndim != 4:
            raise log_error(
                ValueError,
                f"Expected `{attribute}` to have 4 dimensions, "
                f"but got {value.ndim}.",
            )
        if value.shape[-1] not in [2, 3]:
            raise log_error(
                ValueError,
                f"Expected `{attribute}` to have 2 or 3 spatial dimensions, "
                f"but got {value.shape[-1]}.",
            )

    @scores_array.validator
    def _validate_scores_array(self, attribute, value):
        if value is not None:
            _ensure_type_ndarray(value)
            if value.shape != self.tracks_array.shape[:-1]:
                raise log_error(
                    ValueError,
                    f"Expected `{attribute}` to have shape "
                    f"{self.tracks_array.shape[:-1]}, but got {value.shape}.",
                )

    @individual_names.validator
    def _validate_individual_names(self, attribute, value):
        if self.source_software == "LightningPose":
            # LightningPose only supports a single individual
            _validate_list_length(attribute, value, 1)
        else:
            _validate_list_length(attribute, value, self.tracks_array.shape[1])

    @keypoint_names.validator
    def _validate_keypoint_names(self, attribute, value):
        _validate_list_length(attribute, value, self.tracks_array.shape[2])

    def __attrs_post_init__(self):
        """Assign default values to optional attributes (if None)"""
        if self.scores_array is None:
            self.scores_array = np.full(
                (self.tracks_array.shape[:-1]), np.nan, dtype="float32"
            )
            log_warning(
                "Scores array was not provided. Setting to an array of NaNs."
            )
        if self.individual_names is None:
            self.individual_names = [
                f"individual_{i}" for i in range(self.tracks_array.shape[1])
            ]
            log_warning(
                "Individual names were not provided. "
                f"Setting to {self.individual_names}."
            )
        if self.keypoint_names is None:
            self.keypoint_names = [
                f"keypoint_{i}" for i in range(self.tracks_array.shape[2])
            ]
            log_warning(
                "Keypoint names were not provided. "
                f"Setting to {self.keypoint_names}."
            )
