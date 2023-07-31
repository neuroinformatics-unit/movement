import os
from pathlib import Path
from typing import Literal, Optional

import h5py
from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
)


class ValidFile(BaseModel):
    """Pydantic class for validating file paths.

    It ensures that:
    - the path can be converted to a pathlib.Path object.
    - the path does not point to a directory.
    - the file has the expected access permission(s): 'r', 'w', or 'rw'.
    - the file exists `expected_permission` is 'r' or 'rw'.
    - the file does not exist when `expected_permission` is 'w'.
    - the file has one of the expected suffixes, if specified.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the file.
    expected_permission : {'r', 'w', 'rw'}
        Expected access permission(s) for the file. If 'r', the file is
        expected to be readable. If 'w', the file is expected to be writable.
        If 'rw', the file is expected to be both readable and writable.
        Default: 'r'.
    expected_suffix : list of str or None
        Expected suffix(es) for the file. If None (default), this check is
        skipped.

    """

    path: Path
    expected_permission: Literal["r", "w", "rw"] = Field(default="r")
    expected_suffix: Optional[list[str]] = Field(default=None)

    @field_validator("path", mode="before")  # run before instantiation
    def convert_to_path(cls, value):
        if not isinstance(value, Path):
            try:
                value = Path(value)
            except TypeError as error:
                raise error
        return value

    @field_validator("path")
    def path_exists(cls, value):
        if not value.exists():
            raise FileNotFoundError(f"File not found: {value}.")
        return value

    @field_validator("path")
    def path_is_not_dir(cls, value):
        if value.is_dir():
            raise ValueError(
                f"Expected a file but got a directory: {value}. "
                "Please specify a file path."
            )
        return value

    @model_validator(mode="after")
    def file_has_expected_permission(self) -> "ValidFile":
        """Ensure that the file has the expected permission."""
        is_readable = os.access(self.path, os.R_OK)
        is_writeable = os.access(self.path.parent, os.W_OK)

        if self.expected_permission == "r":
            if not is_readable:
                raise PermissionError(
                    f"Unable to read file: {self.path}. "
                    "Make sure that you have read permissions for it."
                )
        elif self.expected_permission == "w":
            if not is_writeable:
                raise PermissionError(
                    f"Unable to write to file: {self.path}. "
                    "Make sure that you have write permissions for it."
                )
        elif self.expected_permission == "rw":
            if not (is_readable and is_writeable):
                raise PermissionError(
                    f"Unable to read and/or write to file: {self.path}. Make"
                    "sure that you have read and write permissions for it."
                )
        return self

    @model_validator(mode="after")
    def file_exists_when_expected(self) -> "ValidFile":
        """Ensure that the file exists when expected (matches the expected
        permission, i.e. the intended use of the file)."""
        if self.expected_permission in ["r", "rw"]:
            if not self.path.exists():
                raise FileNotFoundError(
                    f"Expected file {self.path} does not exist."
                )
        else:
            if self.path.exists():
                raise FileExistsError(
                    f"Expected file {self.path} already exists."
                )
        return self

    @model_validator(mode="after")
    def file_has_expected_suffix(self) -> "ValidFile":
        """Ensure that the file has the expected suffix."""
        if self.expected_suffix is not None:
            if self.path.suffix.lower() not in self.expected_suffix:
                raise ValueError(
                    f"Expected file extension(s) {self.expected_suffix} "
                    f"but got {self.path.suffix} for file: {self.path}."
                )
        return self


class ValidHDF5(BaseModel):
    """Pydantic class for validating HDF5 files. This class ensures that the
    file is a properly formatted and contains the expected datasets
    (if specified).

    Parameters
    ----------
    file : movement.io.validators.ValidFile
        Validated path to the HDF5 file.
    expected_datasets : list of str or None
        List of names of the expected datasets in the HDF5 file. If None
        (default), this check is skipped.
    """

    file: ValidFile
    expected_datasets: Optional[list[str]] = Field(default=None)

    @field_validator("file")
    def file_is_h5(cls, value):
        """Ensure that the file is indeed in HDF5 format."""
        try:
            with h5py.File(value.path, "r") as f:
                assert isinstance(
                    f, h5py.File
                ), f"Expected an HDF5 file but got {type(f)}: {value.path}. "
        except OSError as error:
            raise error
        return value

    @model_validator(mode="after")
    def h5_file_contains_expected_datasets(self) -> "ValidHDF5":
        """Ensure that the HDF5 file contains the expected datasets."""
        if self.expected_datasets is not None:
            with h5py.File(self.file.path, "r") as f:
                diff = set(self.expected_datasets).difference(set(f.keys()))
                print(diff)
                if len(diff) > 0:
                    raise ValueError(
                        f"Could not find the expected dataset(s) {diff} "
                        f"in file: {self.file.path}. "
                    )
        return self


class ValidPosesCSV(BaseModel):
    """Pydantic class for validating CSV files that contain pose estimation
    outputs in DeepLabCut format. This class ensures that the CSV file contains
    the expected index column levels among its top rows.

    Parameters
    ----------
    file : movement.io.validators.ValidFile
        Validated path to the CSV file.
    multianimal : bool
        Whether to ensure that the CSV file contains pose estimation outputs
        for multiple animals. Default: False.
    """

    file: ValidFile
    multianimal: bool = Field(default=False)

    @model_validator(mode="after")
    def csv_file_contains_expected_levels(self) -> "ValidPosesCSV":
        expected_levels = ["scorer", "bodyparts", "coords"]
        if self.multianimal:
            expected_levels.insert(1, "individuals")

        with open(self.file.path, "r") as f:
            header_rows_start = [f.readline().split(",")[0] for _ in range(4)]
            level_in_header_row_starts = [
                level in header_rows_start for level in expected_levels
            ]
            if not all(level_in_header_row_starts):
                raise ValueError(
                    f"The header rows of the CSV file {self.file.path} do not "
                    "contain all expected index column levels "
                    f"{expected_levels}."
                )
        return self
