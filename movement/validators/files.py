"""``attrs`` classes for validating file paths."""

import os
import re
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import orjson
import pandas as pd
from attrs import define, field, validators
from pynwb import NWBFile

from movement.utils.logging import logger

DEFAULT_FRAME_REGEXP = r"(0\d*)\.\w+$"


@define
class ValidFile:
    """Class for validating file paths.

    The validator ensures that the file:

    - is not a directory,
    - exists if it is meant to be read,
    - does not exist if it is meant to be written,
    - has the expected access permission(s), and
    - has one of the expected suffix(es).

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
        If the file does not exist when ``expected_permission`` is "r" or "rw".
    FileExistsError
        If the file exists when ``expected_permission`` is "w".
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
            raise logger.error(
                IsADirectoryError(
                    f"Expected a file path but got a directory: {value}."
                )
            )

    @path.validator
    def _file_exists_when_expected(self, attribute, value):
        """Ensure that the file exists (or not) as needed.

        This depends on the expected usage (read and/or write).
        """
        if "r" in self.expected_permission:
            if not value.exists():
                raise logger.error(
                    FileNotFoundError(f"File {value} does not exist.")
                )
        else:  # expected_permission is "w"
            if value.exists():
                raise logger.error(
                    FileExistsError(f"File {value} already exists.")
                )

    @path.validator
    def _file_has_access_permissions(self, attribute, value):
        """Ensure that the file has the expected access permission(s).

        Raises a PermissionError if not.
        """
        file_is_readable = os.access(value, os.R_OK)
        parent_is_writeable = os.access(value.parent, os.W_OK)
        if ("r" in self.expected_permission) and (not file_is_readable):
            raise logger.error(
                PermissionError(
                    f"Unable to read file: {value}. "
                    "Make sure that you have read permissions."
                )
            )
        if ("w" in self.expected_permission) and (not parent_is_writeable):
            raise logger.error(
                PermissionError(
                    f"Unable to write to file: {value}. "
                    "Make sure that you have write permissions."
                )
            )

    @path.validator
    def _file_has_expected_suffix(self, attribute, value):
        """Ensure that the file has one of the expected suffix(es)."""
        if self.expected_suffix and value.suffix not in self.expected_suffix:
            raise logger.error(
                ValueError(
                    f"Expected file with suffix(es) {self.expected_suffix} "
                    f"but got suffix {value.suffix} instead."
                )
            )


def _validate_file_path(
    file_path: str | Path, expected_suffix: list[str]
) -> ValidFile:
    """Validate the input file path.

    We check that the file has write permission and the expected suffix(es).

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
        logger.error(error)
        raise
    return file


@define
class ValidHDF5:
    """Class for validating HDF5 files.

    The validator ensures that the file:

    - is in HDF5 format, and
    - contains the expected datasets.

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
            raise logger.error(
                ValueError(
                    f"File {value} does not seem to be in validHDF5 format."
                )
            ) from e

    @path.validator
    def _file_contains_expected_datasets(self, attribute, value):
        """Ensure that the HDF5 file contains the expected datasets."""
        if self.expected_datasets:
            with h5py.File(value, "r") as f:
                diff = set(self.expected_datasets).difference(set(f.keys()))
                if len(diff) > 0:
                    raise logger.error(
                        ValueError(
                            f"Could not find the expected dataset(s) {diff} "
                            f"in file: {value}. Make sure that the file "
                            "matches the expected source software format."
                        )
                    )


@define
class ValidDeepLabCutCSV:
    """Class for validating DeepLabCut-style .csv files.

    The validator ensures that the file contains the
    expected index column levels.

    Attributes
    ----------
    path : pathlib.Path
        Path to the .csv file.
    level_names : list of str
        Names of the index column levels found in the .csv file.

    Raises
    ------
    ValueError
        If the .csv file does not contain the expected DeepLabCut index column
        levels among its top rows.

    """

    path: Path = field(validator=validators.instance_of(Path))
    level_names: list[str] = field(init=False, factory=list)

    @path.validator
    def _file_contains_expected_levels(self, attribute, value):
        """Ensure that the .csv file contains the expected index column levels.

        These are to be found in the first column of the first four rows.
        """
        expected_levels = ["scorer", "individuals", "bodyparts", "coords"]
        with open(value) as f:
            level_names = [f.readline().split(",")[0] for _ in range(4)]
            if level_names[3].isdigit():
                # if 4th row starts with a digit, assume single-animal DLC file
                # and compare only first 3 rows, removing 'individuals' level
                expected_levels.remove("individuals")
                level_names.pop()
            if level_names != expected_levels:
                raise logger.error(
                    ValueError(
                        ".csv header rows do not match the known format for "
                        "DeepLabCut pose estimation output files."
                    )
                )
            self.level_names = level_names


@define
class ValidAniposeCSV:
    """Class for validating Anipose-style 3D pose .csv files.

    The validator ensures that the file contains the
    expected column names in its header (first row).

    Attributes
    ----------
    path : pathlib.Path
        Path to the .csv file.

    Raises
    ------
    ValueError
        If the .csv file does not contain the expected Anipose columns.

    """

    path: Path = field(validator=validators.instance_of(Path))

    @path.validator
    def _file_contains_expected_columns(self, attribute, value):
        """Ensure that the .csv file contains the expected columns."""
        expected_column_suffixes = [
            "_x",
            "_y",
            "_z",
            "_score",
            "_error",
            "_ncams",
        ]
        expected_non_keypoint_columns = [
            "fnum",
            "center_0",
            "center_1",
            "center_2",
            "M_00",
            "M_01",
            "M_02",
            "M_10",
            "M_11",
            "M_12",
            "M_20",
            "M_21",
            "M_22",
        ]

        # Read the first line of the CSV to get the headers
        with open(value) as f:
            columns = f.readline().strip().split(",")

        # Check that all expected headers are present
        if not all(col in columns for col in expected_non_keypoint_columns):
            raise logger.error(
                ValueError(
                    "CSV file is missing some expected columns."
                    f"Expected: {expected_non_keypoint_columns}."
                )
            )

        # For other headers, check they have expected suffixes and base names
        other_columns = [
            col for col in columns if col not in expected_non_keypoint_columns
        ]
        for column in other_columns:
            # Check suffix
            if not any(
                column.endswith(suffix) for suffix in expected_column_suffixes
            ):
                raise logger.error(
                    ValueError(
                        f"Column {column} ends with an unexpected suffix."
                    )
                )
            # Get base name by removing suffix
            base = column.rsplit("_", 1)[0]
            # Check base name has all expected suffixes
            if not all(
                f"{base}{suffix}" in columns
                for suffix in expected_column_suffixes
            ):
                raise logger.error(
                    ValueError(
                        f"Keypoint {base} is missing some expected suffixes."
                        f"Expected: {expected_column_suffixes};"
                        f"Got: {columns}."
                    )
                )


@define
class ValidVIATracksCSV:
    """Class for validating VIA tracks .csv files.

    The validator ensures that the file:

    - contains the expected header,
    - contains valid frame numbers,
    - contains tracked bounding boxes, and
    - defines bounding boxes whose IDs are unique per image file.

    If the file is validated, the bounding boxes data is pre-parsed
    from the input file and added as attributes.

    Attributes
    ----------
    path : pathlib.Path
        Path to the VIA tracks .csv file.
    frame_regexp : str
        Regular expression pattern to extract the frame number from the
        filename. By default, the frame number is expected to be encoded in
        the filename as an integer number led by at least one zero, followed
        by the file extension.
    x : list of float
        List of x coordinates of the tracked bounding boxes.
    y : list of float
        List of y coordinates of the tracked bounding boxes.
    w : list of float
        List of width coordinates of the tracked bounding boxes.
    h : list of float
        List of height coordinates of the tracked bounding boxes.
    ids : list of int
        List of track IDs of the tracked bounding boxes.
    frame_numbers : list of int
        List of frame numbers of the tracked bounding boxes.
    confidence_values : list of float
        List of confidence values of the tracked bounding boxes.

    Raises
    ------
    ValueError
        If the file does not match the VIA tracks .csv file requirements.

    """

    path: Path = field(validator=validators.instance_of(Path))
    frame_regexp: str = DEFAULT_FRAME_REGEXP

    # Bboxes pre-parsed data
    df: pd.DataFrame = field(
        init=False, factory=pd.DataFrame
    )  # this dataframe is deleted after validation
    x: list[float] = field(init=False, factory=list)
    y: list[float] = field(init=False, factory=list)
    w: list[float] = field(init=False, factory=list)
    h: list[float] = field(init=False, factory=list)
    ids: list[int] = field(init=False, factory=list)
    frame_numbers: list[int] = field(init=False, factory=list)
    confidence_values: list[float] = field(init=False, factory=list)

    def __attrs_post_init__(self):
        """Clear the dataframe attribute after validation is complete."""
        object.__setattr__(self, "df", None)

    @path.validator
    def _file_contains_valid_header(self, attribute, value):
        """Ensure the VIA tracks .csv file contains the expected header."""
        # Read CSV once and store for later use
        df = pd.read_csv(value, sep=",", header=0)
        expected_header = [
            "filename",
            "file_size",
            "file_attributes",
            "region_count",
            "region_id",
            "region_shape_attributes",
            "region_attributes",
        ]

        if list(df.columns) != expected_header:
            raise logger.error(
                ValueError(
                    ".csv header row does not match the known format for "
                    "VIA tracks .csv files. "
                    f"Expected {expected_header} but got {list(df.columns)}."
                )
            )

        # Store dataframe for other validation steps
        # (deleted once validation is complete)
        self.df = df

    @path.validator
    def _file_contains_valid_frame_numbers(self, attribute, value):
        """Ensure that the VIA tracks .csv file contains valid frame numbers.

        This involves:

        - Checking that frame numbers are included in ``file_attributes`` or
          encoded in the image file ``filename``.
        - Checking the frame number can be cast as an integer.
        - Checking that there are as many unique frame numbers as unique image
          files.

        If the frame number is included as part of the image file name, then
        it is expected to be captured by the regular expression in the
        `frame_regexp` attribute of the ValidVIATracksCSV object. The default
        regexp matches an integer led by at least one zero, followed by the
        file extension.

        """
        # Try extracting frame number from the file attributes
        # (returns None if not defined)
        frame_numbers = []
        for row in self.df["file_attributes"]:
            frame_numbers.append(orjson.loads(row).get("frame"))

        # If there is any None in the list, try extracting
        # the frame number from the filename
        if None in frame_numbers:
            # Check if the regexp pattern can be compiled
            try:
                compiled_pattern = re.compile(self.frame_regexp)
            except re.error as e:
                raise logger.error(
                    ValueError(
                        "The provided regular expression for "
                        "the frame numbers "
                        f"({self.frame_regexp}) could not be compiled. "
                        "Please review its syntax."
                    )
                ) from e

            # Check if the regexp pattern is ill-defined
            if compiled_pattern.groups != 1:
                raise ValueError(
                    "The regexp pattern must contain exactly one capture "
                    f"group for the frame number (got {self.frame_regexp})."
                )

            # Extract frame number from filename
            frame_numbers = self.df["filename"].str.extract(
                self.frame_regexp,
                expand=False,  # to return a series if one capture group
            )

            # Check if there are no matches
            if frame_numbers.isna().any():
                raise logger.error(
                    ValueError(
                        "Could not extract frame numbers from the filenames "
                        f"using the regular expression {self.frame_regexp}. "
                        "Please ensure filenames match the expected pattern, "
                        "or define the frame numbers in file_attributes."
                    )
                )

        # Check all frame numbers are castable as integer
        try:
            frame_numbers_int = []
            for f in frame_numbers:
                frame_numbers_int.append(int(f))
            frame_numbers = frame_numbers_int
        except ValueError as e:
            raise logger.error(
                ValueError(
                    f"Extracted frame number '{f}' cannot be cast as integer. "
                    "Please review the VIA-tracks .csv file."
                )
            ) from e

        # Check we have as many unique frame numbers as unique image files
        if len(set(frame_numbers)) != self.df["filename"].nunique():
            raise logger.error(
                ValueError(
                    "The number of unique frame numbers does not match "
                    "the number of unique image files. Please review the "
                    "VIA tracks .csv file and ensure a unique frame number "
                    "is defined for each file. "
                )
            )

        # If all checks pass, add as attribute
        self.frame_numbers = frame_numbers  # already cast as integer

    @path.validator
    def _file_contains_tracked_bboxes(self, attribute, value):
        """Ensure that the VIA tracks .csv contains tracked bounding boxes.

        This involves:

        - Checking that the bounding boxes are defined as rectangles.
        - Checking that the bounding boxes have all geometric parameters
          (``["x", "y", "width", "height"]``).
        - Checking that the bounding boxes have a track ID defined.
        - Checking that the track ID can be cast as an integer.
        """
        # Extract region shape data
        x, y, w, h, ids, confidence_values = [], [], [], [], [], []
        for k, (shape_row, attr_row) in enumerate(
            zip(
                self.df["region_shape_attributes"],
                self.df["region_attributes"],
                strict=True,
            )
        ):
            # Parse dicts
            shape_attrs = orjson.loads(shape_row)
            region_attrs = orjson.loads(attr_row)

            # Get shape data
            shape_name = shape_attrs.get("name")
            sx, sy, sw, sh = (
                shape_attrs.get("x"),
                shape_attrs.get("y"),
                shape_attrs.get("width"),
                shape_attrs.get("height"),
            )

            # Get ID data
            track_id = region_attrs.get("track")

            # Get confidence data if present (otherwise nan)
            confidence = region_attrs.get("confidence", np.nan)

            # Throw error if invalid shape
            if shape_name != "rect":
                raise logger.error(
                    ValueError(
                        f"The bounding box in row {k + 1} shape was "
                        "expected to be 'rect' (rectangular) but instead got "
                        f"{shape_name}. Please review the VIA tracks "
                        ".csv file."
                    )
                )

            # Throw error if ID is missing
            if track_id is None:
                raise logger.error(
                    ValueError(
                        f"The bounding box in row {k + 1} is "
                        "missing a track ID. "
                        "Please review the VIA tracks .csv file."
                    )
                )

            # Throw error if missing geometry
            if sx is None or sy is None or sw is None or sh is None:
                raise logger.error(
                    ValueError(
                        f"The bounding box in row {k + 1} is "
                        "missing a geometric "
                        "parameter (x, y, width, height). Please review the "
                        "VIA tracks .csv file."
                    )
                )

            # Check if ID is castable as int
            try:
                track_id = int(track_id)
            except ValueError as e:
                raise logger.error(
                    ValueError(
                        f"The track ID of the bounding box in row {k + 1} "
                        "cannot be cast as an integer "
                        f"(got track ID '{track_id}'). Please "
                        "review the VIA tracks .csv file."
                    )
                ) from e

            # Append values to list
            x.append(sx)
            y.append(sy)
            w.append(sw)
            h.append(sh)
            ids.append(track_id)
            confidence_values.append(confidence)

        # If all checks pass, add relevant lists as attributes
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.ids = ids  # already an integer
        self.confidence_values = confidence_values  # nan if not defined

    @path.validator
    def _file_contains_unique_track_ids_per_filename(self, attribute, value):
        """Ensure the VIA tracks .csv contains unique track IDs per filename.

        It checks that bounding boxes IDs are defined once per image file.
        """
        # Use a temporary Series for the check, don't modify self.df
        has_duplicates = self.df.assign(ID=self.ids).duplicated(
            subset=["filename", "ID"],
            keep=False,
        )
        if has_duplicates.any():
            problem_files = (
                self.df.loc[has_duplicates, "filename"].unique().tolist()
            )
            raise logger.error(
                ValueError(
                    "Duplicate track IDs found in the following files: "
                    f"{problem_files}. "
                    "Please review the VIA tracks .csv file."
                )
            )


@define
class ValidNWBFile:
    """Class for validating NWB files.

    The validator ensures that the file is either:

    - a valid NWB file (.nwb) path, or
    - an :class:`NWBFile<pynwb.file.NWBFile>` object.

    Attributes
    ----------
    file : str | Path | pynwb.file.NWBFile
        Path to the NWB file on disk (ending in ".nwb"),
        or an NWBFile object.

    """

    file: str | Path | NWBFile = field(
        converter=lambda f: Path(f) if isinstance(f, str | Path) else f,
        validator=validators.instance_of((Path, NWBFile)),
    )

    @file.validator
    def _file_is_valid_nwb(self, attribute, value):
        """Ensure the file path has read permission and .nwb suffix."""
        if isinstance(value, Path):
            ValidFile(value, expected_permission="r", expected_suffix=[".nwb"])
