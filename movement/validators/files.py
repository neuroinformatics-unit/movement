"""``attrs`` classes for validating file paths."""

import ast
import os
import re
from pathlib import Path
from typing import Literal

import h5py
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

    Raises
    ------
    ValueError
        If the .csv file does not contain the expected DeepLabCut index column
        levels among its top rows.

    """

    path: Path = field(validator=validators.instance_of(Path))

    @path.validator
    def _file_contains_expected_levels(self, attribute, value):
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
                raise logger.error(
                    ValueError(
                        ".csv header rows do not match the known format for "
                        "DeepLabCut pose estimation output files."
                    )
                )


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

    Attributes
    ----------
    path : pathlib.Path
        Path to the VIA tracks .csv file.
    frame_regexp : str
        Regular expression pattern to extract the frame number from the
        filename. By default, the frame number is expected to be encoded in
        the filename as an integer number led by at least one zero, followed
        by the file extension.

    Raises
    ------
    ValueError
        If the file does not match the VIA tracks .csv file requirements.

    """

    path: Path = field(validator=validators.instance_of(Path))
    frame_regexp: str = DEFAULT_FRAME_REGEXP

    @path.validator
    def _file_contains_valid_header(self, attribute, value):
        """Ensure the VIA tracks .csv file contains the expected header."""
        expected_header = [
            "filename",
            "file_size",
            "file_attributes",
            "region_count",
            "region_id",
            "region_shape_attributes",
            "region_attributes",
        ]

        with open(value) as f:
            header = f.readline().strip("\n").split(",")

            if header != expected_header:
                raise logger.error(
                    ValueError(
                        ".csv header row does not match the known format for "
                        "VIA tracks .csv files. "
                        f"Expected {expected_header} but got {header}."
                    )
                )

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
        df = pd.read_csv(value, sep=",", header=0)

        # Extract list of file attributes (dicts)
        file_attributes_dicts = [
            ast.literal_eval(d) for d in df.file_attributes
        ]

        # If 'frame' is a file_attribute for all files:
        # extract frame number
        if all(["frame" in d for d in file_attributes_dicts]):
            list_frame_numbers = (
                self._extract_frame_numbers_from_file_attributes(
                    df, file_attributes_dicts
                )
            )
        # else: extract frame number from filename.
        else:
            list_frame_numbers = self._extract_frame_numbers_using_regexp(df)

        # Check we have as many unique frame numbers as unique image files
        if len(set(list_frame_numbers)) != len(df.filename.unique()):
            raise logger.error(
                ValueError(
                    "The number of unique frame numbers does not match "
                    "the number of unique image files. Please review the "
                    "VIA tracks .csv file and ensure a unique frame number "
                    "is defined for each file. "
                )
            )

    def _extract_frame_numbers_from_file_attributes(
        self, df, file_attributes_dicts
    ):
        """Get frame numbers from the 'frame' key under 'file_attributes'."""
        list_frame_numbers = []
        for k_i, k in enumerate(file_attributes_dicts):
            try:
                list_frame_numbers.append(int(k["frame"]))
            except ValueError as e:
                raise logger.error(
                    ValueError(
                        f"{df.filename.iloc[k_i]} (row {k_i}): "
                        "'frame' file attribute cannot be cast as an integer. "
                        f"Please review the file attributes: {k}."
                    )
                ) from e
        return list_frame_numbers

    def _extract_frame_numbers_using_regexp(self, df):
        """Get frame numbers from the file names using the provided regexp."""
        list_frame_numbers = []
        for f_i, f in enumerate(df["filename"]):
            # try compiling the frame regexp
            try:
                regex_match = re.search(self.frame_regexp, f)
            except re.error as e:
                raise logger.error(
                    re.error(
                        "The provided regular expression for the frame "
                        f"numbers ({self.frame_regexp}) could not be compiled."
                        " Please review its syntax."
                    )
                ) from e
            # try extracting the frame number from the filename using the
            # compiled regexp
            try:
                list_frame_numbers.append(int(regex_match.group(1)))
            except AttributeError as e:
                raise logger.error(
                    AttributeError(
                        f"{f} (row {f_i}): The provided frame regexp "
                        f"({self.frame_regexp}) did not "
                        "return any matches and a frame number could not "
                        "be extracted from the filename."
                    )
                ) from e
            except ValueError as e:
                raise logger.error(
                    ValueError(
                        f"{f} (row {f_i}): "
                        "The frame number extracted from the filename using "
                        f"the provided regexp ({self.frame_regexp}) could not "
                        "be cast as an integer."
                    )
                ) from e

        return list_frame_numbers

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
        df = pd.read_csv(value, sep=",", header=0)

        for row in df.itertuples():
            row_region_shape_attrs = ast.literal_eval(
                row.region_shape_attributes
            )
            row_region_attrs = ast.literal_eval(row.region_attributes)

            # check annotation is a rectangle
            if row_region_shape_attrs["name"] != "rect":
                raise logger.error(
                    ValueError(
                        f"{row.filename} (row {row.Index}): "
                        "bounding box shape must be 'rect' (rectangular) "
                        "but instead got "
                        f"'{row_region_shape_attrs['name']}'."
                    )
                )

            # check all geometric parameters for the box are defined
            if not all(
                [
                    key in row_region_shape_attrs
                    for key in ["x", "y", "width", "height"]
                ]
            ):
                raise logger.error(
                    ValueError(
                        f"{row.filename} (row {row.Index}): "
                        "missing bounding box shape parameter(s). "
                        "Expected 'x', 'y', 'width', 'height' to exist as "
                        "'region_shape_attributes', but got "
                        f"'{list(row_region_shape_attrs.keys())}'."
                    )
                )

            # check track ID is defined
            if "track" not in row_region_attrs:
                raise logger.error(
                    ValueError(
                        f"{row.filename} (row {row.Index}): "
                        "bounding box does not have a 'track' attribute "
                        "defined under 'region_attributes'. "
                        "Please review the VIA tracks .csv file."
                    )
                )

            # check track ID is castable as an integer
            try:
                int(row_region_attrs["track"])
            except Exception as e:
                raise logger.error(
                    ValueError(
                        f"{row.filename} (row {row.Index}): "
                        "the track ID for the bounding box cannot be cast as "
                        "an integer. Please review the VIA tracks .csv file."
                    )
                ) from e

    @path.validator
    def _file_contains_unique_track_ids_per_filename(self, attribute, value):
        """Ensure the VIA tracks .csv contains unique track IDs per filename.

        It checks that bounding boxes IDs are defined once per image file.
        """
        df = pd.read_csv(value, sep=",", header=0)

        list_unique_filenames = list(set(df.filename))
        for file in list_unique_filenames:
            df_one_filename = df.loc[df["filename"] == file]

            list_track_ids_one_filename = [
                int(ast.literal_eval(row.region_attributes)["track"])
                for row in df_one_filename.itertuples()
            ]

            if len(set(list_track_ids_one_filename)) != len(
                list_track_ids_one_filename
            ):
                raise logger.error(
                    ValueError(
                        f"{file}: "
                        "multiple bounding boxes in this file "
                        "have the same track ID. "
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
