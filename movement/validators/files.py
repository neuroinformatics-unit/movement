"""``attrs`` classes for validating file paths."""

import os
from pathlib import Path
from typing import Literal

import h5py
from attrs import define, field, validators

from movement.utils.logging import log_error


@define
class ValidFile:
    """Class for validating file paths.

    Attributes
    ----------
    path : str or pathlib.Path
        Path to the file.
    expected_permission : {"r", "w", "rw"}
        Expected access permission(s) for the file. If "r", the file is
        expected to be readable. If "w", the file is expected to be writable.
        If "rw", the file is expected to be both readable and writable.
        Default: "r".
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
        If the file does not exist when `expected_permission` is "r" or "rw".
    FileExistsError
        If the file exists when `expected_permission` is "w".
    ValueError
        If the file does not have one of the expected suffix(es).

    """

    path: Path = field(converter=Path, validator=validators.instance_of(Path))
    expected_permission: Literal["r", "w", "rw"] = field(
        default="r", validator=validators.in_(["r", "w", "rw"]), kw_only=True
    )
    expected_suffix: list[str] = field(factory=list, kw_only=True)

    @path.validator
    def _path_is_not_dir(self, attribute, value):
        """Ensure that the path does not point to a directory."""
        if value.is_dir():
            raise log_error(
                IsADirectoryError,
                f"Expected a file path but got a directory: {value}.",
            )

    @path.validator
    def _file_exists_when_expected(self, attribute, value):
        """Ensure that the file exists (or not) as needed.

        This depends on the expected usage (read and/or write).
        """
        if "r" in self.expected_permission:
            if not value.exists():
                raise log_error(
                    FileNotFoundError, f"File {value} does not exist."
                )
        else:  # expected_permission is "w"
            if value.exists():
                raise log_error(
                    FileExistsError, f"File {value} already exists."
                )

    @path.validator
    def _file_has_access_permissions(self, attribute, value):
        """Ensure that the file has the expected access permission(s).

        Raises a PermissionError if not.
        """
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
    def _file_has_expected_suffix(self, attribute, value):
        """Ensure that the file has one of the expected suffix(es)."""
        if self.expected_suffix and value.suffix not in self.expected_suffix:
            raise log_error(
                ValueError,
                f"Expected file with suffix(es) {self.expected_suffix} "
                f"but got suffix {value.suffix} instead.",
            )


@define
class ValidHDF5:
    """Class for validating HDF5 files.

    Attributes
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
    def _file_is_h5(self, attribute, value):
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
    def _file_contains_expected_datasets(self, attribute, value):
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
class ValidDeepLabCutCSV:
    """Class for validating DeepLabCut-style .csv files.

    Attributes
    ----------
    path : pathlib.Path
        Path to the .csv file.

    Raises
    ------
    ValueError
        If the .csv file does not contain the expected DeepLabCut index column
        levels among its top rows.

    """

    path: Path = field(validator=validators.instance_of(Path))

    @path.validator
    def _csv_file_contains_expected_levels(self, attribute, value):
        """Ensure that the .csv file contains the expected index column levels.

        These are to be found among the top 4 rows of the file.
        """
        expected_levels = ["scorer", "bodyparts", "coords"]

        with open(value) as f:
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
                    ".csv header rows do not match the known format for "
                    "DeepLabCut pose estimation output files.",
                )
