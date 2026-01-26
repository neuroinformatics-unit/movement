import stat
from pathlib import Path

import pytest

from movement.validators.files import (
    ValidAniposeCSV,
    ValidDeepLabCutCSV,
    ValidFile,
    ValidHDF5,
    ValidNWBFile,
    ValidROICollectionGeoJSON,
    ValidVIATracksCSV,
    _validate_file_path,
)
from movement.validators.json_schemas import get_schema


@pytest.fixture
def sample_file_path():
    """Return a factory of file paths with a given file extension suffix."""

    def _sample_file_path(tmp_path: Path, suffix: str):
        """Return a path for a file under the pytest temporary directory
        with the given file extension.
        """
        file_path = tmp_path / f"test.{suffix}"
        return file_path

    return _sample_file_path


@pytest.mark.parametrize("suffix", [".txt", ".csv"])
def test_validate_file_path_valid_file(sample_file_path, tmp_path, suffix):
    """Test file path validation with a correct file."""
    file_path = sample_file_path(tmp_path, suffix)
    validated_file = _validate_file_path(file_path, [suffix])

    assert isinstance(validated_file, ValidFile)
    assert validated_file.path == file_path


@pytest.mark.parametrize("suffix", [".txt", ".csv"])
def test_validate_file_path_invalid_permission(
    sample_file_path, tmp_path, suffix
):
    """Test file path validation with a file that has invalid permissions.

    We use the following permissions:
    - S_IRUSR: Read permission for owner
    - S_IRGRP: Read permission for group
    - S_IROTH: Read permission for others
    """
    # Create a sample file with read-only permission
    file_path = sample_file_path(tmp_path, suffix)
    file_path.touch()
    file_path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)

    # Try to validate the file path
    # (should raise an OSError since we require write permissions)
    with pytest.raises(OSError):
        _validate_file_path(file_path, [suffix])


@pytest.mark.parametrize("suffix", [".txt", ".csv"])
def test_validate_file_path_file_exists(sample_file_path, tmp_path, suffix):
    """Test file path validation with a file that already exists.

    We use the following permissions to create a file with the right
    permissions:
    - S_IRUSR: Read permission for owner
    - S_IWUSR: Write permission for owner
    - S_IRGRP: Read permission for group
    - S_IWGRP: Write permission for group
    - S_IROTH: Read permission for others
    - S_IWOTH: Write permission for others

    We include both read and write permissions because in real-world
    scenarios it's very rare to have a file that is writable but not readable.
    """
    # Create a sample file with read and write permissions
    file_path = sample_file_path(tmp_path, suffix)
    file_path.touch()
    file_path.chmod(
        stat.S_IRUSR
        | stat.S_IWUSR
        | stat.S_IRGRP
        | stat.S_IWGRP
        | stat.S_IROTH
        | stat.S_IWOTH
    )

    # Try to validate the file path
    # (should raise an OSError since the file already exists)
    with pytest.raises(OSError):
        _validate_file_path(file_path, [suffix])


@pytest.mark.parametrize("invalid_suffix", [".foo", "", None])
def test_validate_file_path_invalid_suffix(
    sample_file_path, tmp_path, invalid_suffix
):
    """Test file path validation with an invalid file suffix."""
    # Create a file path with an invalid suffix
    file_path = sample_file_path(tmp_path, invalid_suffix)

    # Try to validate using a .txt suffix
    with pytest.raises(ValueError):
        _validate_file_path(file_path, [".txt"])


@pytest.mark.parametrize("suffix", [".txt", ".csv"])
def test_validate_file_path_multiple_suffixes(
    sample_file_path, tmp_path, suffix
):
    """Test file path validation with multiple valid suffixes."""
    # Create a valid txt file path
    file_path = sample_file_path(tmp_path, suffix)

    # Validate using multiple valid suffixes
    validated_file = _validate_file_path(file_path, [".txt", ".csv"])

    assert isinstance(validated_file, ValidFile)
    assert validated_file.path == file_path


@pytest.mark.parametrize(
    "invalid_input, expected_exception",
    [
        ("unreadable_file", pytest.raises(PermissionError)),
        ("unwriteable_file", pytest.raises(PermissionError)),
        ("fake_h5_file", pytest.raises(FileExistsError)),
        ("wrong_extension_file", pytest.raises(ValueError)),
        ("nonexistent_file", pytest.raises(FileNotFoundError)),
        ("directory", pytest.raises(IsADirectoryError)),
    ],
)
def test_file_validator_with_invalid_input(
    invalid_input, expected_exception, request
):
    """Test that invalid files raise the appropriate errors."""
    invalid_dict = request.getfixturevalue(invalid_input)
    with expected_exception:
        ValidFile(
            invalid_dict.get("file_path"),
            expected_permission=invalid_dict.get("expected_permission"),
            expected_suffix=invalid_dict.get("expected_suffix", []),
        )


@pytest.mark.parametrize(
    "invalid_input, expected_exception",
    [
        ("no_dataframe_h5_file", pytest.raises(ValueError)),
        ("fake_h5_file", pytest.raises(ValueError)),
    ],
)
def test_hdf5_validator_with_invalid_input(
    invalid_input, expected_exception, request
):
    """Test that invalid HDF5 files raise the appropriate errors."""
    invalid_dict = request.getfixturevalue(invalid_input)
    with expected_exception:
        ValidHDF5(
            invalid_dict.get("file_path"),
            expected_datasets=invalid_dict.get("expected_datasets"),
        )


@pytest.mark.parametrize(
    "invalid_input, expected_exception",
    [
        ("invalid_single_individual_csv_file", pytest.raises(ValueError)),
        ("invalid_multi_individual_csv_file", pytest.raises(ValueError)),
    ],
)
def test_deeplabcut_csv_validator_with_invalid_input(
    invalid_input, expected_exception, request
):
    """Test that invalid CSV files raise the appropriate errors."""
    file_path = request.getfixturevalue(invalid_input)
    with expected_exception:
        ValidDeepLabCutCSV(file_path)


@pytest.mark.parametrize(
    "invalid_input, error_type, log_message",
    [
        (
            "via_invalid_header",
            ValueError,
            ".csv header row does not match the known format for "
            "VIA tracks .csv files. "
            "Expected "
            "['filename', 'file_size', 'file_attributes', "
            "'region_count', 'region_id', 'region_shape_attributes', "
            "'region_attributes'] "
            "but got ['filename', 'file_size', 'file_attributes'].",
        ),
        (
            "via_frame_number_in_file_attribute_not_integer",
            ValueError,
            "04.09.2023-04-Right_RE_test_frame_A.png (row 0): "
            "'frame' file attribute cannot be cast as an integer. "
            "Please review the file attributes: "
            "{'clip': 123, 'frame': 'FOO'}.",
        ),
        (
            "via_frame_number_in_filename_wrong_pattern",
            AttributeError,
            "04.09.2023-04-Right_RE_test_frame_1.png (row 0): "
            r"The provided frame regexp ((0\d*)\.\w+$) did not return "
            "any matches and a "
            "frame number could not be extracted from the "
            "filename.",
        ),
        (
            "via_more_frame_numbers_than_filenames",
            ValueError,
            "The number of unique frame numbers does not match the number "
            "of unique image files. Please review the VIA tracks .csv file "
            "and ensure a unique frame number is defined for each file. ",
        ),
        (
            "via_less_frame_numbers_than_filenames",
            ValueError,
            "The number of unique frame numbers does not match the number "
            "of unique image files. Please review the VIA tracks .csv file "
            "and ensure a unique frame number is defined for each file. ",
        ),
        (
            "via_region_shape_attribute_not_rect",
            ValueError,
            "04.09.2023-04-Right_RE_test_frame_01.png (row 0): "
            "bounding box shape must be 'rect' (rectangular) "
            "but instead got 'circle'.",
        ),
        (
            "via_region_shape_attribute_missing_x",
            ValueError,
            "04.09.2023-04-Right_RE_test_frame_01.png (row 0): "
            "missing bounding box shape parameter(s). "
            "Expected 'x', 'y', 'width', 'height' to exist as "
            "'region_shape_attributes', but got "
            "'['name', 'y', 'width', 'height']'.",
        ),
        (
            "via_region_attribute_missing_track",
            ValueError,
            "04.09.2023-04-Right_RE_test_frame_01.png (row 0): "
            "bounding box does not have a 'track' attribute defined "
            "under 'region_attributes'. "
            "Please review the VIA tracks .csv file.",
        ),
        (
            "via_track_id_not_castable_as_int",
            ValueError,
            "04.09.2023-04-Right_RE_test_frame_01.png (row 0): "
            "the track ID for the bounding box cannot be cast "
            "as an integer. "
            "Please review the VIA tracks .csv file.",
        ),
        (
            "via_track_ids_not_unique_per_frame",
            ValueError,
            "04.09.2023-04-Right_RE_test_frame_01.png: "
            "multiple bounding boxes in this file have the same track ID. "
            "Please review the VIA tracks .csv file.",
        ),
    ],
)
def test_via_tracks_csv_validator_with_invalid_input(
    invalid_via_tracks_csv_file, invalid_input, error_type, log_message
):
    """Test that invalid VIA tracks .csv files raise the appropriate errors.

    Errors to check:
    - error if .csv header is wrong
    - error if frame number is not defined in the file
        (frame number extracted either from the filename or from attributes)
    - error if extracted frame numbers are not 1-based integers
    - error if region_shape_attributes "name" is not "rect"
    - error if not all region_attributes have key "track"
        (i.e., all regions must have an ID assigned)
    - error if IDs are unique per frame
        (i.e., bboxes IDs must exist only once per frame)
    - error if bboxes IDs are not 1-based integers
    """
    file_path = invalid_via_tracks_csv_file(invalid_input)
    with pytest.raises(error_type) as excinfo:
        ValidVIATracksCSV(file_path)

    assert str(excinfo.value) == log_message


@pytest.mark.parametrize(
    "invalid_input, log_message",
    [
        (
            "invalid_single_individual_csv_file",
            "CSV file is missing some expected columns.",
        ),
        (
            "missing_keypoint_columns_anipose_csv_file",
            "Keypoint kp0 is missing some expected suffixes.",
        ),
        (
            "spurious_column_anipose_csv_file",
            "Column funny_column ends with an unexpected suffix.",
        ),
    ],
)
def test_anipose_csv_validator_with_invalid_input(
    invalid_input, log_message, request
):
    """Test that invalid Anipose .csv files raise the appropriate errors.

    Errors to check:
    - error if .csv is missing some columns
    - error if .csv misses some of the expected columns for a keypoint
    - error if .csv has columns that are not expected
    (either common ones or keypoint-specific ones)
    """
    file_path = request.getfixturevalue(invalid_input)
    with pytest.raises(ValueError) as excinfo:
        ValidAniposeCSV(file_path)

    assert log_message in str(excinfo.value)


@pytest.mark.parametrize(
    "invalid_input, expected_exception",
    [
        ("wrong_extension_file", pytest.raises(ValueError)),
        (123, pytest.raises(TypeError)),
        (None, pytest.raises(TypeError)),
    ],
)
def test_nwb_file_validator_with_invalid_input(
    invalid_input, expected_exception, request
):
    """Test that invalid NWB file inputs raise the appropriate errors."""
    with expected_exception:
        invalid_input = (
            request.getfixturevalue(invalid_input).get("file_path")
            if invalid_input == "wrong_extension_file"
            else invalid_input
        )
        ValidNWBFile(invalid_input)


class TestValidROICollectionGeoJSON:
    """Tests for ValidROICollectionGeoJSON validator."""

    def test_valid_feature_collection(self, tmp_path):
        """Test that a valid FeatureCollection passes validation."""
        file_path = tmp_path / "valid.geojson"
        file_path.write_text(
            '{"type": "FeatureCollection", "features": ['
            '{"type": "Feature", "geometry": {"type": "Polygon", '
            '"coordinates": [[[0,0],[1,0],[1,1],[0,0]]]}, "properties": {}}'
            "]}"
        )
        validated = ValidROICollectionGeoJSON(file_path)
        assert validated.path == file_path

    def test_invalid_json(self, tmp_path):
        """Test that invalid JSON raises ValueError."""
        file_path = tmp_path / "invalid.geojson"
        file_path.write_text("not valid json {")

        with pytest.raises(ValueError, match="not valid JSON"):
            ValidROICollectionGeoJSON(file_path)

    def test_not_feature_collection(self, tmp_path):
        """Test that non-FeatureCollection raises ValueError."""
        file_path = tmp_path / "feature.geojson"
        file_path.write_text('{"type": "Feature", "geometry": null}')

        with pytest.raises(ValueError, match="does not match schema"):
            ValidROICollectionGeoJSON(file_path)

    def test_missing_features_key(self, tmp_path):
        """Test that missing 'features' key raises ValueError."""
        file_path = tmp_path / "no_features.geojson"
        file_path.write_text('{"type": "FeatureCollection"}')

        with pytest.raises(
            ValueError, match="'features' is a required property"
        ):
            ValidROICollectionGeoJSON(file_path)

    def test_missing_geometry(self, tmp_path):
        """Test that feature without geometry raises ValueError."""
        file_path = tmp_path / "no_geometry.geojson"
        file_path.write_text(
            '{"type": "FeatureCollection", "features": ['
            '{"type": "Feature", "properties": {}}'
            "]}"
        )

        with pytest.raises(
            ValueError, match="'geometry' is a required property"
        ):
            ValidROICollectionGeoJSON(file_path)

    def test_null_geometry(self, tmp_path):
        """Test that null geometry raises ValueError."""
        file_path = tmp_path / "null_geometry.geojson"
        file_path.write_text(
            '{"type": "FeatureCollection", "features": ['
            '{"type": "Feature", "geometry": null, "properties": {}}'
            "]}"
        )

        with pytest.raises(ValueError, match="None is not of type 'object'"):
            ValidROICollectionGeoJSON(file_path)

    def test_unsupported_geometry_type(self, tmp_path):
        """Test that unsupported geometry type raises ValueError."""
        file_path = tmp_path / "point.geojson"
        file_path.write_text(
            '{"type": "FeatureCollection", "features": ['
            '{"type": "Feature", "geometry": {"type": "Point", '
            '"coordinates": [0, 0]}, "properties": {}}'
            "]}"
        )

        with pytest.raises(
            ValueError, match=r"is not one of \['Polygon', 'LineString'"
        ):
            ValidROICollectionGeoJSON(file_path)

    @pytest.mark.parametrize(
        ["geometry_type", "roi_type"],
        [
            pytest.param(
                "LineString",
                "PolygonOfInterest",
                id="LineString with PolygonOfInterest",
            ),
            pytest.param(
                "Polygon",
                "LineOfInterest",
                id="Polygon with LineOfInterest",
            ),
        ],
    )
    def test_roi_type_geometry_mismatch(
        self, geometry_type, roi_type, tmp_path
    ):
        """Test that roi_type/geometry mismatch raises TypeError."""
        if geometry_type == "Polygon":
            coords = "[[[0,0],[1,0],[1,1],[0,0]]]"
        else:
            coords = "[[0,0],[1,1]]"

        file_path = tmp_path / "mismatch.geojson"
        file_path.write_text(
            f'{{"type": "FeatureCollection", "features": ['
            f'{{"type": "Feature", "geometry": {{"type": "{geometry_type}", '
            f'"coordinates": {coords}}}, '
            f'"properties": {{"roi_type": "{roi_type}"}}}}'
            f"]}}"
        )

        with pytest.raises(TypeError, match="does not match geometry type"):
            ValidROICollectionGeoJSON(file_path)

    def test_unknown_roi_type(self, tmp_path):
        """Test that unknown roi_type raises ValueError."""
        file_path = tmp_path / "unknown_roi_type.geojson"
        file_path.write_text(
            '{"type": "FeatureCollection", "features": ['
            '{"type": "Feature", "geometry": {"type": "Polygon", '
            '"coordinates": [[[0,0],[1,0],[1,1],[0,0]]]}, '
            '"properties": {"roi_type": "UnknownROI"}}'
            "]}"
        )

        with pytest.raises(
            ValueError, match=r"is not one of \['PolygonOfInterest'"
        ):
            ValidROICollectionGeoJSON(file_path)

    def test_empty_feature_collection(self, tmp_path):
        """Test that empty FeatureCollection is valid."""
        file_path = tmp_path / "empty.geojson"
        file_path.write_text('{"type": "FeatureCollection", "features": []}')

        validated = ValidROICollectionGeoJSON(file_path)
        assert validated.path == file_path


class TestGetSchema:
    """Tests for the get_schema utility function."""

    def test_get_schema_existing(self):
        """Test that get_schema loads an existing schema."""
        schema = get_schema("roi_collection")
        assert isinstance(schema, dict)
        assert schema.get("type") == "object"
