"""``attrs`` classes for validating file paths."""

import ast
import os
import re
from pathlib import Path
from typing import Literal

import h5py
import pandas as pd
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


@define
class ValidVIAtracksCSV:
    """Class for validating VIA tracks .csv files.

    Parameters
    ----------
    path : pathlib.Path or str
        Path to the .csv file.

    Raises
    ------
    ValueError
        If the .csv file does not match the VIA tracks file requirements.

    """

    path: Path = field(validator=validators.instance_of(Path))

    @path.validator
    def csv_file_contains_expected_levels(self, attribute, value):
        """Ensure that the .csv file contains the expected VIA tracks columns.

        These should be the header of the file.
        """
        # Check all columns output by VIA (even if we don't use them all)
        expected_levels = [
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

            if header != expected_levels:
                raise log_error(
                    ValueError,
                    ".csv header row does not match the known format for "
                    "VIA tracks output files.",
                )

    @path.validator
    def csv_file_contains_frame_numbers(self, attribute, value):
        """Check csv file contains frame numbers.

        Check if frame number defined in one of: file_attributes OR filename
        (log which one)
        Check frame number is defined for all frames
        Check frame number is a 1-based integer.
        """
        # Read file as dataframe
        df_file = pd.read_csv(value, sep=",", header=0)

        # Extract file attributes
        list_file_attrs = [
            ast.literal_eval(d) for d in df_file.file_attributes
        ]  # list of dicts

        # check all file attributes are the same
        assert all([f == list_file_attrs[0] for f in list_file_attrs[1:]])

        # if frame is defined as a file attribute: extract
        if "frame" in list_file_attrs[0]:
            list_frame_numbers = [
                int(
                    file_attr["frame"]
                )  # what if it cannot be converted to an int? use try?
                for file_attr in list_file_attrs
            ]
        # else: extract from filename
        # frame number is expected between "_" and ".",
        # led by at least one zero, followed by extension
        else:
            pattern = r"_(0\d*)\.\w+$"
            list_frame_numbers = [
                int(re.search(pattern, f).group(1))  # type: ignore
                for f in df_file["filename"]
                if re.search(
                    pattern, f
                )  # only added if there is a pattern match
            ]

        list_unique_frame_numbers = list(set(list_frame_numbers))

        # Check frame numbers are defined for all files
        assert len(list_unique_frame_numbers) == len(set(df_file.filename))

        # Check frame number is a 1-based integer
        # (we enforce that is integer previously)
        assert all(
            [
                f > 0  # and isinstance(f, int)
                for f in list_unique_frame_numbers
            ]
        )

    @path.validator
    def csv_file_contains_boxes(self, attribute, value):
        """Check csv file contains bounding boxes.

        Check region_shape_attributes "name" is "rect"
        otherwise shape width and height doesn't make sense
        """
        # Read file as dataframe
        df_file = pd.read_csv(value, sep=",", header=0)

        for _, row in df_file.iterrows():
            assert (
                ast.literal_eval(row.region_shape_attributes)["name"] == "rect"
            )
            assert "x" in ast.literal_eval(row.region_shape_attributes)
            assert "y" in ast.literal_eval(row.region_shape_attributes)
            assert "width" in ast.literal_eval(row.region_shape_attributes)
            assert "height" in ast.literal_eval(row.region_shape_attributes)

    @path.validator
    def csv_file_contains_1_based_tracks(self, attribute, value):
        """Check csv file contains 1-based track IDs.

        Check all region_attributes have key "track",
        Check track IDs are 1-based integers
        """
        # Read file as dataframe
        df_file = pd.read_csv(value, sep=",", header=0)

        # Extract all bounding boxes IDs
        # as list comprehension?
        list_bbox_ID = []
        for _, row in df_file.iterrows():
            assert "track" in ast.literal_eval(row.region_attributes)

            list_bbox_ID.append(
                int(ast.literal_eval(row.region_attributes)["track"])
            )

        # Check all IDs are 1-based integers
        assert all([f > 0 for f in list_bbox_ID])

    @path.validator
    def csv_file_contains_unique_track_IDs_per_frame(self, attribute, value):
        """Check csv file contains unique track IDs per frame.

        Check bboxes IDs exist only once per frame/file
        """
        # Read csv file as dataframe
        df = pd.read_csv(value, sep=",", header=0)

        # Extract subdataframes grouped by filename (frame)
        list_unique_filenames = list(set(df.filename))
        for file in list_unique_filenames:
            # One dataframe for a filename (frame)
            df_one_file = df.loc[df["filename"] == file]

            # Extract IDs per filename (frame)
            list_bbox_ID_one_file = [
                int(ast.literal_eval(row.region_attributes)["track"])
                for _, row in df_one_file.iterrows()
            ]

            # Check the IDs are unique per frame
            assert len(set(list_bbox_ID_one_file)) == len(
                list_bbox_ID_one_file
            )
