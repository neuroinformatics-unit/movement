"""``attrs`` classes for validating file paths."""

import json
import os
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, ClassVar, Literal, Protocol

import h5py
import jsonschema
import numpy as np
import orjson
import pandas as pd
from attrs import Attribute, define, field, validators
from pynwb import NWBFile

from movement.utils.logging import logger
from movement.validators._json_schemas import (
    ROI_COLLECTION_SCHEMA,
    ROI_TYPE_TO_GEOMETRY,
)

DEFAULT_FRAME_REGEXP = r"(0\d*)\.\w+$"


class ValidFile(Protocol):
    """Protocol for file validation classes."""

    suffixes: ClassVar[set[str]]
    """Expected suffix(es) for the file."""

    file: Path | NWBFile
    """Path to the file to validate or an NWBFile object."""


# --- Composable attrs validators --- #


def _file_validator(
    *,
    permission: Literal["r", "w", "rw"] = "r",
    suffixes: set[str] | None = None,
) -> Callable[[Any, Attribute, Any], Any]:
    """Return a validator that composes file checks.

    The validator ensures that the file:

    - is not a directory,
    - exists if it is meant to be read,
    - does not exist if it is meant to be written,
    - has the expected access permission(s), and
    - has one of the expected suffix(es).

    Parameters
    ----------
    permission
        Expected access permission(s) for the file. If "r", the file is
        expected to be readable. If "w", the file is expected to be writable.
        If "rw", the file is expected to be both readable and writable.
        Default is "r".
    suffixes
        Expected suffix(es) for the file.
        If None (default), this check is skipped.

    Raises
    ------
    TypeError
        If the value is not a :class:`pathlib.Path` object.
    IsADirectoryError
        If the file points to a directory.
    PermissionError
        If the expected access permission(s) are not met.
    FileNotFoundError
        If the file does not exist when ``permission`` is "r" or "rw".
    FileExistsError
        If the file exists when ``permission`` is "w".
    ValueError
        If the file does not have one of the expected suffix(es).

    """
    v = [
        validators.instance_of(Path),
        _file_is_not_dir,
        _file_is_accessible(permission),
    ]
    if suffixes:
        v.append(_file_has_expected_suffix(suffixes))
    return validators.and_(*v)


def _file_is_not_dir(_, __, value: Path) -> None:
    """Ensure the file does not point to a directory."""
    if value.is_dir():
        raise logger.error(
            IsADirectoryError(
                f"Expected a file path but got a directory: {value}."
            )
        )


def _file_is_readable(value: Path) -> None:
    """Ensure the file exists and is readable."""
    if not value.exists():
        raise logger.error(FileNotFoundError(f"File {value} does not exist."))
    if not os.access(value, os.R_OK):
        raise logger.error(
            PermissionError(
                f"Unable to read file: {value}. "
                "Make sure that you have read permissions."
            )
        )


def _file_is_writable(value: Path) -> None:
    """Ensure the file does not exist and parent directory is writable."""
    if value.exists():
        raise logger.error(FileExistsError(f"File {value} already exists."))
    if not os.access(value.parent, os.W_OK):
        raise logger.error(
            PermissionError(
                f"Unable to write to file: {value}. "
                "Make sure that you have write permissions."
            )
        )


def _file_is_accessible(
    expected_permission: Literal["r", "w", "rw"],
) -> Callable[[Any, Any, Path], None]:
    """Ensure the file can be accessed with the expected permission(s)."""
    if expected_permission not in {"r", "w", "rw"}:
        raise logger.error(
            ValueError(
                f"expected_permission must be one of 'r', 'w', or 'rw', "
                f"but got '{expected_permission}' instead."
            )
        )

    def _validator(_, __, value: Path) -> None:
        if "r" in expected_permission:
            _file_is_readable(value)
        if "w" in expected_permission:
            _file_is_writable(value)

    return _validator


def _file_has_expected_suffix(
    suffixes: set[str],
) -> Callable[[Any, Any, Path], None]:
    """Ensure the file has one of the expected suffix(es)."""

    def _validator(_, __, value: Path) -> None:
        if value.suffix not in suffixes:
            raise logger.error(
                ValueError(
                    f"Expected file with suffix(es) {suffixes} "
                    f"but got suffix {value.suffix} instead."
                )
            )

    return _validator


def _hdf5_validator(
    datasets: set[str],
) -> Callable[[Any, Any, Path], None]:
    """Return a validator for HDF5 files.

    The validator ensures that the file:

    - is in HDF5 format, and
    - contains the expected datasets.

    Parameters
    ----------
    datasets
        Set of names of the expected datasets in the HDF5 file.

    Raises
    ------
    ValueError
        If the HDF5 group does not contain the expected datasets.

    """

    def _validator(_, __, value: Path) -> None:
        try:
            with h5py.File(value, "r") as f:
                diff = set(datasets).difference(set(f.keys()))
                if len(diff) > 0:
                    raise logger.error(
                        ValueError(
                            f"Could not find the expected dataset(s) {diff} "
                            f"in file: {value}. Make sure that the file "
                            "matches the expected source software format."
                        )
                    )
        except OSError as e:
            raise logger.error(
                ValueError(
                    f"Could not open file as HDF5: {value}. "
                    f"Make sure that the file is a valid HDF5 file. "
                    f"Error: {e}"
                )
            ) from e

    return _validator


def _json_validator(
    schema: Mapping[str, Any] | None = None,
    custom_checks: tuple[Callable[[Mapping[str, Any]], None], ...] = (),
    data_attr: str | None = None,
) -> Callable[[Any, Any, Path], None]:
    """Return a validator for JSON files.

    The validator ensures that the file:

    - is in valid JSON format, and
    - matches the provided JSON schema, if given.

    Parameters
    ----------
    schema
        The JSON schema to validate against. If None (default), any valid
        JSON file is accepted.
    custom_checks
        Additional custom checks to apply to the JSON data. Each check must
        be a callable that receives the parsed JSON data as its only argument.
    data_attr
        Optional name of an attribute on the validated instance where the
        parsed JSON data should be stored.

    Raises
    ------
    ValueError
        If the file is not valid JSON or does not match the schema.

    """

    def _validator(instance: Any, __: Attribute, value: Path) -> None:
        try:
            with open(value) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise logger.error(
                ValueError(f"File {value} is not valid JSON: {e}")
            ) from e

        if schema is not None:
            try:
                jsonschema.validate(instance=data, schema=schema)
            except jsonschema.ValidationError as e:
                raise logger.error(
                    ValueError(
                        f"File {value} does not match schema: {e.message}"
                    )
                ) from e

        for check in custom_checks:
            check(data)

        if data_attr is not None:
            setattr(instance, data_attr, data)

    return _validator


def _if_instance_of(
    cls: type, validator: Callable[[Any, Attribute, Any], None]
) -> Callable[[Any, Attribute, Any], None]:
    """Return a validator that conditionally applies based on type.

    Use this to apply a validator only when the value is an instance
    of a specific class. Useful for fields that accept union types.

    Parameters
    ----------
    cls
        The class type to check against.
    validator
        The validator to apply if the value is an instance of ``cls``.

    Returns
    -------
    collections.abc.Callable
        A validator function that conditionally applies the given
        validator.

    """

    def _validator(instance: Any, attribute: Attribute, value: Any) -> None:
        if isinstance(value, cls):
            validator(instance, attribute, value)

    return _validator


# --- Helper functions --- #


def validate_file_path(
    file: Path | str,
    *,
    permission: Literal["r", "w", "rw"] = "r",
    suffixes: set[str] | None = None,
) -> Path:
    """Validate the file has the expected permission(s) and suffix(es).

    Parameters
    ----------
    file
        Path to the file to validate.
    permission
        Expected access permission(s) for the file. If "r", the file is
        expected to be readable. If "w", the file is expected to be writable.
        If "rw", the file is expected to be both readable and writable.
        Default is "r".
    suffixes
        Expected suffix(es) for the file. If None (default),
        this check is skipped.

    Returns
    -------
    Path
        The validated file path.

    Raises
    ------
    OSError
        If the file does not meet the expected access ``permission`` or
        if it is a directory.
    ValueError
        If the file does not have one of the expected ``suffixes`` or
        if the ``permission`` argument is invalid.

    """
    try:

        @define
        class _TempValidator:
            file: Path = field(
                converter=Path,
                validator=_file_validator(
                    permission=permission, suffixes=suffixes
                ),
            )

        validator = _TempValidator(file=file)
        return validator.file
    except (OSError, ValueError) as error:
        logger.error(error)
        raise


# --- File validator classes --- #


@define
class ValidSleapAnalysis:
    """Class for validating SLEAP analysis (.h5) files."""

    suffixes: ClassVar[set[str]] = {".h5"}
    """Expected suffix(es) for the file."""

    file: Path = field(
        converter=Path,
        validator=validators.and_(
            _file_validator(permission="r", suffixes=suffixes),
            _hdf5_validator(datasets={"tracks"}),
        ),
    )
    """Path to the SLEAP .h5 file to validate."""


@define
class ValidSleapLabels:
    """Class for validating SLEAP labels (.slp) files."""

    suffixes: ClassVar[set[str]] = {".slp"}
    """Expected suffix(es) for the file."""

    file: Path = field(
        converter=Path,
        validator=validators.and_(
            _file_validator(permission="r", suffixes=suffixes),
            _hdf5_validator(datasets={"pred_points", "metadata"}),
        ),
    )
    """Path to the SLEAP .slp file to validate."""


@define
class ValidDeepLabCutH5:
    """Class for validating DeepLabCut-style .h5 files."""

    suffixes: ClassVar[set[str]] = {".h5"}
    """Expected suffix(es) for the file."""

    file: Path = field(
        converter=Path,
        validator=validators.and_(
            _file_validator(permission="r", suffixes=suffixes),
            _hdf5_validator(datasets={"df_with_missing"}),
        ),
    )
    """Path to the DeepLabCut .h5 file to validate."""


@define
class ValidDeepLabCutCSV:
    """Class for validating DeepLabCut-style .csv files.

    The validator ensures that the file contains the
    expected index column levels.

    Raises
    ------
    ValueError
        If the .csv file does not contain the expected DeepLabCut index column
        levels among its top rows.

    """

    suffixes: ClassVar[set[str]] = {".csv"}
    """Expected suffix(es) for the file."""

    file: Path = field(
        converter=Path,
        validator=_file_validator(permission="r", suffixes=suffixes),
    )
    """Path to the DeepLabCut .csv file to validate."""

    level_names: list[str] = field(init=False, factory=list)
    """Names of the index column levels found in the .csv file."""

    @file.validator
    def _file_contains_expected_levels(self, attribute, value):
        """Ensure that the .csv file contains the expected index column levels.

        These are to be found in the first column of the first four rows.
        """
        expected_levels = ["scorer", "individual", "bodyparts", "coords"]
        legacy_levels = ["scorer", "individuals", "bodyparts", "coords"]
        with open(value) as f:
            level_names = [f.readline().split(",")[0] for _ in range(4)]
            if level_names[3].isdigit():
                # if 4th row starts with a digit, assume single-animal DLC file
                # and compare only first 3 rows, removing 'individual' level
                expected_levels.remove("individual")
                legacy_levels.remove("individuals")
                level_names.pop()
            if level_names != expected_levels and level_names != legacy_levels:
                raise logger.error(
                    ValueError(
                        ".csv header rows do not match the known format for "
                        "DeepLabCut pose estimation output files."
                    )
                )
            self.level_names = [
                "individual" if x == "individuals" else x for x in level_names
            ]


@define
class ValidAniposeCSV:
    """Class for validating Anipose-style 3D pose .csv files.

    The validator ensures that the file contains the
    expected column names in its header (first row).

    Raises
    ------
    ValueError
        If the .csv file does not contain the expected Anipose columns.

    """

    suffixes: ClassVar[set[str]] = {".csv"}
    """Expected suffix(es) for the file."""

    file: Path = field(
        converter=Path,
        validator=_file_validator(permission="r", suffixes=suffixes),
    )
    """Path to the Anipose .csv file to validate."""

    @file.validator
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

    Raises
    ------
    ValueError
        If the file does not match the VIA tracks .csv file requirements.

    """

    suffixes: ClassVar[set[str]] = {".csv"}
    """Expected suffix(es) for the file."""

    file: Path = field(
        converter=Path,
        validator=_file_validator(permission="r", suffixes=suffixes),
    )
    """Path to the VIA tracks .csv file to validate."""

    frame_regexp: str = field(default=DEFAULT_FRAME_REGEXP)
    """Regular expression pattern to extract the frame number from the
    filename. By default, the frame number is expected to be encoded in
    the filename as an integer number led by at least one zero, followed
    by the file extension."""

    # Bboxes pre-parsed data
    x: np.typing.NDArray[np.float32] = field(init=False)
    """Array of x coordinates of the tracked bounding boxes."""

    y: np.typing.NDArray[np.float32] = field(init=False)
    """Array of y coordinates of the tracked bounding boxes."""

    w: np.typing.NDArray[np.float32] = field(init=False)
    """Array of width coordinates of the tracked bounding boxes."""

    h: np.typing.NDArray[np.float32] = field(init=False)
    """Array of height coordinates of the tracked bounding boxes."""

    ids: np.typing.NDArray[np.int32] = field(init=False)
    """Array of track IDs of the tracked bounding boxes."""

    frame_numbers: np.typing.NDArray[np.int64] = field(init=False)
    """Array of frame numbers corresponding to the tracked bounding boxes."""

    confidence: np.typing.NDArray[np.float32] = field(init=False)
    """Array of confidence values of the tracked bounding boxes."""

    @file.validator
    def _validate_via_tracks_file(self, attribute, value):
        """Run all VIA tracks validations and cache parsed attributes."""
        # Validate frame_regexp first since file
        # validation depends on it
        self._frame_regexp_valid(self.frame_regexp)

        # Read csv as a dataframe
        df = pd.read_csv(value)

        # Run all validation steps
        self._file_contains_valid_header(df)
        frame_numbers = self._file_contains_valid_frame_numbers(df)
        x, y, w, h, ids, confidence_values = (
            self._file_contains_tracked_bboxes(df)
        )
        self._file_contains_unique_track_ids_per_filename(df, ids)

        # If all checks pass, add parsed attributes to the object
        self.frame_numbers = frame_numbers
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.ids = ids
        self.confidence = confidence_values

    @frame_regexp.validator
    def _validate_frame_regexp(self, attribute, value):
        """Validate the frame_regexp attribute.

        It also runs when the frame_regexp field is reassigned.
        """
        self._frame_regexp_valid(value)

    def _frame_regexp_valid(self, value):
        """Ensure the frame regexp pattern is valid.

        Checks regexp pattern can be compiled and that it contains
        exactly one capture group.
        """
        # Check if the regexp pattern can be compiled
        try:
            compiled_pattern = re.compile(value)
        except re.error as e:
            raise logger.error(
                ValueError(
                    "The provided regular expression for "
                    "the frame numbers "
                    f"({value}) could not be compiled. "
                    "Please review its syntax."
                )
            ) from e

        # Check it contains one capture group
        if compiled_pattern.groups != 1:
            raise ValueError(
                "The regexp pattern must contain exactly one capture "
                f"group for the frame number (got {value})."
            )

    def _file_contains_valid_header(self, df: pd.DataFrame):
        """Ensure the VIA tracks .csv file contains the expected header."""
        # Read CSV once and store for later use
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

    def _file_contains_valid_frame_numbers(
        self, df: pd.DataFrame
    ) -> np.ndarray:
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
        frame_numbers = pd.Series(
            [orjson.loads(row).get("frame") for row in df["file_attributes"]]
        )

        # If there is any None in the list, try extracting
        # the frame number from the filename
        if frame_numbers.isna().any():
            # Extract frame number from filename
            frame_numbers = df["filename"].str.extract(
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
            frame_numbers = frame_numbers.astype(int)
        except (ValueError, TypeError) as e:
            raise logger.error(
                ValueError(
                    "Some frame numbers cannot be cast as integer. "
                    "Please review the VIA-tracks .csv file."
                )
            ) from e

        # Check we have as many unique frame numbers as unique image files
        if frame_numbers.nunique() != df["filename"].nunique():
            raise logger.error(
                ValueError(
                    "The number of unique frame numbers does not match "
                    "the number of unique image files. Please review the "
                    "VIA tracks .csv file and ensure a unique frame number "
                    "is defined for each file. "
                )
            )

        # If all checks pass, return
        return np.asarray(frame_numbers.values)  # already cast as integer

    def _file_contains_tracked_bboxes(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, ...]:
        """Ensure that the VIA tracks .csv contains tracked bounding boxes.

        This involves:

        - Checking that the bounding boxes are defined as rectangles.
        - Checking that the bounding boxes have all geometric parameters
          (``["x", "y", "width", "height"]``).
        - Checking that the bounding boxes have a track ID defined.
        - Checking that the track ID can be cast as an integer.
        """
        # Extract region shape data
        n_rows = len(df)
        x = np.empty(n_rows, dtype=np.float32)
        y = np.empty(n_rows, dtype=np.float32)
        w = np.empty(n_rows, dtype=np.float32)
        h = np.empty(n_rows, dtype=np.float32)
        ids = np.empty(n_rows, dtype=np.int32)
        confidence_values = np.full(n_rows, np.nan, dtype=np.float32)

        for k, (shape_row, attr_row) in enumerate(
            zip(
                df["region_shape_attributes"],
                df["region_attributes"],
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
                        f"The bounding box shape in row {k + 1} is "
                        "expected to be 'rect' (rectangular) but instead got "
                        f"{shape_name}. Please review the VIA tracks "
                        ".csv file."
                    )
                )

            # Throw error if missing geometry
            if None in (sx, sy, sw, sh):
                raise logger.error(
                    ValueError(
                        f"The bounding box in row {k + 1} is "
                        "missing a geometric "
                        "parameter (x, y, width, height). Please review the "
                        "VIA tracks .csv file."
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
            x[k] = sx
            y[k] = sy
            w[k] = sw
            h[k] = sh
            ids[k] = track_id
            confidence_values[k] = confidence

        # If all checks pass, return lists
        # ids is already cast as integer, confidence_values is nan
        # if not defined
        return x, y, w, h, ids, confidence_values

    def _file_contains_unique_track_ids_per_filename(
        self, df: pd.DataFrame, ids: np.ndarray
    ):
        """Ensure the VIA tracks .csv contains unique track IDs per filename.

        It checks that bounding boxes IDs are defined once per image file.
        """
        # Use a temporary series for the check by using `.assign`
        # (so that we don't modify self.df)
        has_duplicates = df.assign(ID=ids).duplicated(
            subset=["filename", "ID"],
            keep=False,
        )
        if has_duplicates.any():
            problem_files = (
                df.loc[has_duplicates, "filename"].unique().tolist()
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
    """

    suffixes: ClassVar[set[str]] = {".nwb"}
    """Expected suffix(es) for the file."""

    file: Path | NWBFile = field(
        converter=lambda f: Path(f) if isinstance(f, str | Path) else f,
        validator=validators.and_(
            validators.instance_of((Path, NWBFile)),
            _if_instance_of(
                Path,
                _file_validator(permission="r", suffixes=suffixes),
            ),
        ),
    )
    """Path to the NWB file on disk (ending in ".nwb") or an NWBFile object."""


def _check_roi_type_matches_geometry(data: Mapping[str, Any]) -> None:
    """Ensure ``roi_type`` properties match the GeoJSON geometry types.

    This custom ValidROICollectionGeoJSON check enforces that the ``roi_type``
    properties in the GeoJSON file are consistent with their geometry types,
    according to the ``ROI_TYPE_TO_GEOMETRY`` mapping.
    """
    for i, feature in enumerate(data.get("features", [])):
        roi_type = feature.get("properties", {}).get("roi_type")

        if roi_type is None:
            continue

        geom_type = feature.get("geometry", {}).get("type")
        expected_geom_types = ROI_TYPE_TO_GEOMETRY.get(roi_type, ())

        if geom_type not in expected_geom_types:
            raise logger.error(
                TypeError(
                    f"Feature {i}: roi_type '{roi_type}' "
                    f"does not match geometry type "
                    f"'{geom_type}'"
                )
            )


@define
class ValidROICollectionGeoJSON:
    """Class for validating GeoJSON FeatureCollection files.

    The validator ensures that the file:

    - is well-formed JSON.
    - conforms to the RoI Collection GeoJSON schema, which checks that
      the file contains a GeoJSON FeatureCollection containing
      ``movement``-compatible regions of interest (as produced by
      :func:`save_rois()<movement.roi.save_rois>`).

    Additionally, it performs a custom validation step to ensure that
    each Feature's ``roi_type`` property is consistent with its geometry
    type (e.g. "PolygonOfInterest" must have geometry type "Polygon").

    Raises
    ------
    ValueError
        If the file is not valid JSON or does not match the expected schema.
    TypeError
        If ``roi_type`` property does not match the actual geometry type.

    See Also
    --------
    movement.roi.save_rois : Save a collection of RoIs to a GeoJSON file.
    movement.roi.load_rois : Load a collection of RoIs from a GeoJSON file.

    """

    suffixes: ClassVar[set[str]] = {".geojson", ".json"}
    """Expected suffix(es) for the file."""

    schema: ClassVar[Mapping[str, Any]] = ROI_COLLECTION_SCHEMA
    """JSON schema for validating the structure of the GeoJSON file."""

    file: Path = field(
        converter=Path,
        validator=validators.and_(
            _file_validator(permission="r", suffixes=suffixes),
            _json_validator(
                schema=schema,
                custom_checks=(_check_roi_type_matches_geometry,),
                data_attr="data",
            ),
        ),
    )
    """Path to the GeoJSON file to validate."""

    data: dict = field(init=False, factory=dict)
    """Parsed JSON data from the file, available after validation."""
