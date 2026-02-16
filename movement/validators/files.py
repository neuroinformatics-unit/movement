"""``attrs`` classes for validating file paths."""

import ast
import json
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar, Literal, Protocol

import h5py
import jsonschema
import pandas as pd
from attrs import Attribute, define, field, validators
from pynwb import NWBFile

from movement.utils.logging import logger

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
    schema: dict | None = None,
) -> Callable[[Any, Any, Path], None]:
    """Return a validator for JSON files.

    The validator ensures that the file:

    - is in valid JSON format, and
    - matches the provided JSON schema, if given.

    Parameters
    ----------
    schema
        The JSON schema to validate against. If None (default), only the JSON
        format will be checked.

    Raises
    ------
    ValueError
        If the file is not valid JSON or does not match the schema.

    """

    def _validator(_, __, value: Path) -> None:
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

    return _validator


def _get_json_schema(name: str) -> dict:
    """Load a JSON schema from file.

    The ``name`` argument should correspond to the name of a JSON schema file
    (without the .json extension) located in the ``json_schemas`` directory
    """
    schema_path = Path(__file__).parent / "json_schemas" / f"{name}.json"
    with open(schema_path) as file:
        return json.load(file)


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
    Callable
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
    file: Path = field(
        converter=Path,
        validator=validators.and_(
            _file_validator(permission="r", suffixes=suffixes),
            _hdf5_validator(datasets={"tracks"}),
        ),
    )


@define
class ValidSleapLabels:
    """Class for validating SLEAP labels (.slp) files."""

    suffixes: ClassVar[set[str]] = {".slp"}
    file: Path = field(
        converter=Path,
        validator=validators.and_(
            _file_validator(permission="r", suffixes=suffixes),
            _hdf5_validator(datasets={"pred_points", "metadata"}),
        ),
    )


@define
class ValidDeepLabCutH5:
    """Class for validating DeepLabCut-style .h5 files."""

    suffixes: ClassVar[set[str]] = {".h5"}
    file: Path = field(
        converter=Path,
        validator=validators.and_(
            _file_validator(permission="r", suffixes=suffixes),
            _hdf5_validator(datasets={"df_with_missing"}),
        ),
    )


@define
class ValidDeepLabCutCSV:
    """Class for validating DeepLabCut-style .csv files.

    The validator ensures that the file contains the
    expected index column levels.

    Attributes
    ----------
    file
        Path to the .csv file.
    level_names
        Names of the index column levels found in the .csv file.

    Raises
    ------
    ValueError
        If the .csv file does not contain the expected DeepLabCut index column
        levels among its top rows.

    """

    suffixes: ClassVar[set[str]] = {".csv"}
    file: Path = field(
        converter=Path,
        validator=_file_validator(permission="r", suffixes=suffixes),
    )
    level_names: list[str] = field(init=False, factory=list)

    @file.validator
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
    file
        Path to the .csv file.

    Raises
    ------
    ValueError
        If the .csv file does not contain the expected Anipose columns.

    """

    suffixes: ClassVar[set[str]] = {".csv"}
    file: Path = field(
        converter=Path,
        validator=_file_validator(permission="r", suffixes=suffixes),
    )

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

    Attributes
    ----------
    file
        Path to the VIA tracks .csv file.
    frame_regexp
        Regular expression pattern to extract the frame number from the
        filename. By default, the frame number is expected to be encoded in
        the filename as an integer number led by at least one zero, followed
        by the file extension.

    Raises
    ------
    ValueError
        If the file does not match the VIA tracks .csv file requirements.

    """

    suffixes: ClassVar[set[str]] = {".csv"}
    file: Path = field(
        converter=Path,
        validator=_file_validator(permission="r", suffixes=suffixes),
    )
    frame_regexp: str = field(default=DEFAULT_FRAME_REGEXP)

    @file.validator
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

    @file.validator
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

    @file.validator
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

    @file.validator
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
    file
        Path to the NWB file on disk (ending in ".nwb"),
        or an NWBFile object.

    """

    suffixes: ClassVar[set[str]] = {".nwb"}
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


@define
class ValidROICollectionGeoJSON:
    """Class for validating GeoJSON FeatureCollection files.

    The validator ensures that the file is:

    - in valid JSON format.
    - matches the expected ``roi_collection`` JSON schema. This schema captures
      the structure of a GeoJSON file containing a FeatureCollection
      of ``movement``-compatible regions of interest.
      (as produced by :func:`save_rois()<movement.roi.save_rois>`).

    Additionally, a custom check is implemented to ensure that
    each Feature's ``roi_type`` property matches its actual geometry type
    (e.g., "PolygonOfInterest" should have geometry type "Polygon").

    Attributes
    ----------
    file
        Path to the GeoJSON file.
    data
        Parsed JSON data from the file.

    Raises
    ------
    ValueError
        If the file is not valid JSON or does not match the expected schema.
    TypeError
        If roi_type property does not match the actual geometry type.

    See Also
    --------
    movement.roi.save_rois : Save a collection of RoIs to a GeoJSON file.
    movement.roi.load_rois : Load a collection of RoIs from a GeoJSON file.

    """

    suffixes: ClassVar[set[str]] = {".geojson", ".json"}
    schema: ClassVar[dict] = _get_json_schema("roi_collection")
    roi_type_to_geometry: ClassVar[dict[str, tuple[str, ...]]] = {
        "PolygonOfInterest": ("Polygon",),
        "LineOfInterest": ("LineString", "LinearRing"),
    }
    file: Path = field(
        converter=Path,
        validator=validators.and_(
            _file_validator(permission="r", suffixes=suffixes),
            _json_validator(schema=schema),
        ),
    )
    data: dict = field(init=False, factory=dict)

    def __attrs_post_init__(self):
        """Load data and check roi_type/geometry consistency."""
        with open(self.file) as f:
            self.data = json.load(f)
        self._check_roi_type_matches_geometry()

    def _check_roi_type_matches_geometry(self):
        """Ensure roi_type properties match actual geometry types.

        This cross-field validation cannot be expressed in JSON Schema,
        so it's implemented here as a post-initialisation custom check.
        """
        for i, feature in enumerate(self.data.get("features", [])):
            properties = feature.get("properties", {})
            roi_type = properties.get("roi_type") if properties else None

            if roi_type is None:
                continue

            geometry = feature.get("geometry", {})
            geom_type = geometry.get("type") if geometry else None
            expected_geom_types = self.roi_type_to_geometry.get(roi_type)

            if geom_type not in expected_geom_types:
                raise logger.error(
                    TypeError(
                        f"Feature {i}: roi_type '{roi_type}' "
                        f"does not match geometry type "
                        f"'{geom_type}'"
                    )
                )
