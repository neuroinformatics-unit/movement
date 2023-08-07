import logging
import os
from pathlib import Path
from typing import List, Literal

import h5py
from attrs import define, field, validators

# get logger
logger = logging.getLogger(__name__)


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
    expected_suffix: List[str] = field(factory=list, kw_only=True)

    @path.validator
    def path_is_not_dir(self, attribute, value):
        """Ensures that the path does not point to a directory."""
        if value.is_dir():
            raise IsADirectoryError(
                f"Expected a file path but got a directory: {value}."
            )

    @path.validator
    def file_exists_when_expected(self, attribute, value):
        """Ensures that the file exists (or not) depending on the expected
        usage (read and/or write)."""
        if "r" in self.expected_permission:
            if not value.exists():
                raise FileNotFoundError(f"File {value} does not exist.")
        else:  # expected_permission is 'w'
            if value.exists():
                raise FileExistsError(f"File {value} already exists.")

    @path.validator
    def file_has_access_permissions(self, attribute, value):
        """Ensures that the file has the expected access permission(s).
        Raises a PermissionError if not."""
        if "r" in self.expected_permission:
            if not os.access(value, os.R_OK):
                raise PermissionError(
                    f"Unable to read file: {value}. "
                    "Make sure that you have read permissions for it."
                )
        if "w" in self.expected_permission:
            if not os.access(value.parent, os.W_OK):
                raise PermissionError(
                    f"Unable to write to file: {value}. "
                    "Make sure that you have write permissions for it."
                )

    @path.validator
    def file_has_expected_suffix(self, attribute, value):
        """Ensures that the file has one of the expected suffix(es)."""
        if self.expected_suffix:  # list is not empty
            if value.suffix not in self.expected_suffix:
                raise ValueError(
                    f"Expected file with suffix(es) {self.expected_suffix} "
                    f"but got suffix {value.suffix} instead."
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
    expected_datasets: List[str] = field(factory=list, kw_only=True)

    @path.validator
    def file_is_h5(self, attribute, value):
        """Ensure that the file is indeed in HDF5 format."""
        try:
            with h5py.File(value, "r") as f:
                f.close()
        except Exception as e:
            raise ValueError(
                f"File {value} does not seem to be in valid" "HDF5 format."
            ) from e

    @path.validator
    def file_contains_expected_datasets(self, attribute, value):
        """Ensure that the HDF5 file contains the expected datasets."""
        if self.expected_datasets:
            with h5py.File(value, "r") as f:
                diff = set(self.expected_datasets).difference(set(f.keys()))
                if len(diff) > 0:
                    raise ValueError(
                        f"Could not find the expected dataset(s) {diff} "
                        f"in file: {value}. "
                    )


@define
class ValidPosesCSV:
    """Class for validating CSV files that contain pose estimation outputs.
    in DeepLabCut format.

    Parameters
    ----------
    path : pathlib.Path
        Path to the CSV file.
    multianimal : bool
        Whether to ensure that the CSV file contains pose estimation outputs
        for multiple animals. Default: False.

    Raises
    ------
    ValueError
        If the CSV file does not contain the expected DeepLabCut index column
        levels among its top rows.
    """

    path: Path = field(validator=validators.instance_of(Path))
    multianimal: bool = field(default=False, kw_only=True)

    @path.validator
    def csv_file_contains_expected_levels(self, attribute, value):
        """Ensure that the CSV file contains the expected index column levels
        among its top rows."""
        expected_levels = ["scorer", "bodyparts", "coords"]
        if self.multianimal:
            expected_levels.insert(1, "individuals")

        with open(value, "r") as f:
            header_rows_start = [f.readline().split(",")[0] for _ in range(4)]
            level_in_header_row_starts = [
                level in header_rows_start for level in expected_levels
            ]
            if not all(level_in_header_row_starts):
                raise ValueError(
                    f"The header rows of the CSV file {value} do not "
                    "contain all expected index column levels "
                    f"{expected_levels}."
                )
