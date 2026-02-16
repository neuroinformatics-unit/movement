import re
from contextlib import nullcontext as does_not_raise
from pathlib import Path
from unittest.mock import Mock

import pytest
from attrs import define, field

from movement.validators.files import (
    ValidAniposeCSV,
    ValidDeepLabCutCSV,
    ValidDeepLabCutH5,
    ValidNWBFile,
    ValidROICollectionGeoJSON,
    ValidSleapAnalysis,
    ValidSleapLabels,
    ValidVIATracksCSV,
    _hdf5_validator,
    _if_instance_of,
    validate_file_path,
)


@pytest.mark.parametrize(
    "input, permission, suffixes, expected_context",
    [
        ("readable_csv_file", "r", None, does_not_raise()),
        ("readable_csv_file", "r", {".csv"}, does_not_raise()),
        ("readable_csv_file", "r", {".csv", ".h5"}, does_not_raise()),
        ("new_csv_file", "w", None, does_not_raise()),
        ("unreadable_file", "r", None, pytest.raises(PermissionError)),
        ("unwriteable_file", "w", None, pytest.raises(PermissionError)),
        ("fake_h5_file", "w", None, pytest.raises(FileExistsError)),
        (
            "wrong_extension_file",
            "r",
            {".h5", ".csv"},
            pytest.raises(ValueError),
        ),
        ("nonexistent_file", "r", None, pytest.raises(FileNotFoundError)),
        ("directory", "r", None, pytest.raises(IsADirectoryError)),
        ("new_csv_file", "x", None, pytest.raises(ValueError)),
    ],
    ids=[
        "has read permission, exists, and is not a directory",
        "has expected suffix",
        "has one of the expected suffixes",
        "has write permission and does not exist",
        "lacks read permission",
        "lacks write permission",
        "write permission is expected, but file already exists",
        "invalid file suffix",
        "read permission is expected, but file does not exist",
        "path is a directory",
        "invalid expected permission",
    ],
)
def test_validate_file_path(
    input, permission, suffixes, expected_context, request
):
    """Test `validate_file_path` and the underlying `_file_validator`.
    If input is valid, the returned value is a Path object, otherwise
    the appropriate error is raised.
    """
    file_path = request.getfixturevalue(input)
    with expected_context:
        validated_file = validate_file_path(
            file_path,
            permission=permission,
            suffixes=suffixes,
        )
        assert isinstance(validated_file, Path)


@pytest.mark.parametrize(
    "input, expected_datasets, expected_context",
    [
        (
            "data_as_list_h5_file",
            {"dataframe"},
            pytest.raises(
                ValueError, match="Could not find the expected dataset"
            ),
        ),
        (
            "fake_h5_file",
            set(),
            pytest.raises(ValueError, match="Could not open file as HDF5"),
        ),
        ("data_as_list_h5_file", {"data_as_list"}, does_not_raise()),
    ],
)
def test_hdf5_validator(input, expected_datasets, expected_context, request):
    """Test `_hdf5_validator` with valid and invalid inputs."""

    @define
    class _StubValidator:
        file: Path = field(
            converter=Path,
            validator=_hdf5_validator(datasets=expected_datasets),
        )

    with expected_context:
        _StubValidator(file=request.getfixturevalue(input))


@pytest.mark.parametrize(
    "value, validator_should_be_called",
    [
        (1, True),
        (1.00, False),
    ],
)
def test_if_instance_of(value, validator_should_be_called):
    """Test the `_if_instance_of` validator.

    The validator should only apply the mocked validator if the value
    is an instance of the specified type (int in this case).
    """
    mock_validator = Mock()

    @define
    class _StubValidator:
        value: int | float = field(
            validator=_if_instance_of(int, mock_validator)
        )

    _StubValidator(value=value)
    if validator_should_be_called:
        mock_validator.assert_called_once()
    else:
        mock_validator.assert_not_called()


@pytest.mark.parametrize(
    "validator_cls, file_fixture, expected_context",
    [
        (ValidSleapAnalysis, "sleap_analysis_file", does_not_raise()),
        (
            ValidSleapAnalysis,
            "sleap_slp_file",
            pytest.raises(ValueError, match="Expected file with suffix"),
        ),
        (
            ValidSleapAnalysis,
            "dlc_h5_file",
            pytest.raises(
                ValueError, match="Could not find the expected dataset"
            ),
        ),
        (ValidSleapLabels, "sleap_slp_file", does_not_raise()),
        (
            ValidSleapLabels,
            "sleap_analysis_file",
            pytest.raises(ValueError, match="Expected file with suffix"),
        ),
    ],
    ids=[
        "Analysis file validator used with SLEAP analysis file",
        "Analysis file validator used with SLEAP labels file",
        "Analysis file validator used with DeepLabCut .h5 file",
        "Labels file validator used with SLEAP labels file",
        "Labels file validator used with SLEAP analysis file",
    ],
)
def test_sleap_validators(
    validator_cls, file_fixture, expected_context, request
):
    """Test SLEAP validators with valid and invalid inputs."""
    file = request.getfixturevalue(file_fixture)
    with expected_context:
        validator_cls(file)


@pytest.mark.parametrize(
    "validator_cls, file_fixture, expected_context",
    [
        (ValidDeepLabCutCSV, "dlc_csv_file", does_not_raise()),
        (
            ValidDeepLabCutCSV,
            "dlc_h5_file",
            pytest.raises(ValueError, match="Expected file with suffix"),
        ),
        (
            ValidDeepLabCutCSV,
            "invalid_single_individual_csv_file",
            pytest.raises(ValueError, match="header rows do not match"),
        ),
        (
            ValidDeepLabCutCSV,
            "invalid_multi_individual_csv_file",
            pytest.raises(ValueError, match="header rows do not match"),
        ),
        (ValidDeepLabCutH5, "dlc_h5_file", does_not_raise()),
        (
            ValidDeepLabCutH5,
            "dlc_csv_file",
            pytest.raises(ValueError, match="Expected file with suffix"),
        ),
        (
            ValidDeepLabCutH5,
            "sleap_analysis_file",
            pytest.raises(
                ValueError, match="Could not find the expected dataset"
            ),
        ),
    ],
    ids=[
        "CSV file validator used with DLC .csv file",
        "CSV file validator used with DLC .h5 file",
        "Invalid single-individual DLC .csv file",
        "Invalid multi-individual DLC .csv file",
        "H5 file validator used with DLC .h5 file",
        "H5 file validator used with DLC .csv file",
        "H5 file validator used with SLEAP analysis file",
    ],
)
def test_deeplabcut_validators(
    validator_cls, file_fixture, expected_context, request
):
    """Test DeepLabCut validators with valid and invalid inputs."""
    file = request.getfixturevalue(file_fixture)
    with expected_context:
        validator_cls(file)


@pytest.mark.parametrize(
    "mode, param, expected_context",
    [
        (
            "file",
            "via_invalid_header",
            pytest.raises(ValueError, match=".csv header row does not match"),
        ),
        (
            "file",
            "via_frame_number_in_file_attribute_not_integer",
            pytest.raises(
                ValueError,
                match="'frame' file attribute cannot be cast as an integer",
            ),
        ),
        (
            "file",
            "via_frame_number_in_filename_wrong_pattern",
            pytest.raises(
                AttributeError,
                match="provided frame regexp .* did not return any matches",
            ),
        ),
        (
            "file",
            "via_more_frame_numbers_than_filenames",
            pytest.raises(
                ValueError,
                match="number of unique frame numbers does not match",
            ),
        ),
        (
            "file",
            "via_less_frame_numbers_than_filenames",
            pytest.raises(
                ValueError,
                match="number of unique frame numbers does not match",
            ),
        ),
        (
            "file",
            "via_region_shape_attribute_not_rect",
            pytest.raises(
                ValueError, match="bounding box shape must be 'rect'"
            ),
        ),
        (
            "file",
            "via_region_shape_attribute_missing_x",
            pytest.raises(
                ValueError, match="missing bounding box shape parameter"
            ),
        ),
        (
            "file",
            "via_region_attribute_missing_track",
            pytest.raises(
                ValueError,
                match="bounding box does not have a 'track' attribute",
            ),
        ),
        (
            "file",
            "via_track_id_not_castable_as_int",
            pytest.raises(
                ValueError,
                match="the track ID for the bounding box cannot be cast",
            ),
        ),
        (
            "file",
            "via_track_ids_not_unique_per_frame",
            pytest.raises(
                ValueError,
                match="multiple bounding boxes .* have the same track ID",
            ),
        ),
        (
            "regexp",
            r"*",
            pytest.raises(re.error, match="could not be compiled"),
        ),
        (
            "regexp",
            r"_(0\d*)_$",
            pytest.raises(
                AttributeError,
                match="provided frame regexp .* did not return any matches",
            ),
        ),
        (
            "regexp",
            r"(0\d*\.\w+)$",
            pytest.raises(
                ValueError,
                match="frame number .* could not be cast as an integer",
            ),
        ),
        ("valid", None, does_not_raise()),
    ],
)
def test_via_tracks_csv_validator(
    invalid_via_tracks_csv_file, via_tracks_csv, mode, param, expected_context
):
    """Test ValidVIATracksCSV with valid and invalid inputs.

    Errors to check
    - file errors
        - .csv header is wrong
        - frame number is not defined in the file
          (frame number extracted either from the filename or from attributes)
        - extracted frame numbers cannot be cast as integers
        - region_shape_attributes "name" is not "rect"
        - not all region_attributes have key "track"
          (i.e., all regions must have an ID assigned)
        - IDs are unique per frame
          (i.e., bboxes IDs must exist only once per frame)
        - bboxes IDs cannot be cast as integers
    - invalid frame_regexp
        - regexp cannot be compiled
        - regexp does not return any matches
        - extracted frame numbers cannot be cast as integers
    """
    file_path = (
        invalid_via_tracks_csv_file(param)
        if mode == "file"
        else via_tracks_csv
    )
    with expected_context:
        if mode != "regexp":
            ValidVIATracksCSV(file_path)
        else:
            ValidVIATracksCSV(file_path, frame_regexp=param)


@pytest.mark.parametrize(
    "input, expected_context",
    [
        (
            "invalid_single_individual_csv_file",
            pytest.raises(ValueError, match="missing some expected columns"),
        ),
        (
            "missing_keypoint_columns_anipose_csv_file",
            pytest.raises(ValueError, match="missing some expected suffixes"),
        ),
        (
            "spurious_column_anipose_csv_file",
            pytest.raises(ValueError, match="ends with an unexpected suffix"),
        ),
        ("anipose_csv_file", does_not_raise()),
    ],
)
def test_anipose_csv_validator(input, expected_context, request):
    """Test ValidAniposeCSV with valid and invalid inputs.

    Errors to check:
    - .csv is missing some columns
    - .csv misses some of the expected columns for a keypoint
    - .csv has columns that are not expected (common or keypoint-specific)
    """
    file_path = request.getfixturevalue(input)
    with expected_context:
        ValidAniposeCSV(file_path)


@pytest.mark.parametrize(
    "input, expected_context",
    [
        ("nwb_file", does_not_raise()),
        ("nwbfile_object", does_not_raise()),
        (
            "dlc_csv_file",
            pytest.raises(ValueError, match="Expected file with suffix"),
        ),
    ],
)
def test_nwb_file_validator(input, expected_context, request):
    """Test ValidNWBFile with valid and invalid inputs."""
    file = request.getfixturevalue(input)
    if input.startswith("nwb"):
        file = file()
    with expected_context:
        ValidNWBFile(file)


_POLYGON_FEATURE = (
    '{"type": "Feature", "geometry": {"type": "Polygon", '
    '"coordinates": [[[0,0],[1,0],[1,1],[0,0]]]}, "properties": {}}'
)


def _feature_collection(*features: str) -> str:
    """Build a GeoJSON FeatureCollection string."""
    joined = ", ".join(features)
    return f'{{"type": "FeatureCollection", "features": [{joined}]}}'


def _feature_with_roi_type(geom_type: str, coords: str, roi_type: str) -> str:
    """Build a GeoJSON Feature string with an roi_type property."""
    return (
        f'{{"type": "Feature", '
        f'"geometry": {{"type": "{geom_type}", '
        f'"coordinates": {coords}}}, '
        f'"properties": {{"roi_type": "{roi_type}"}}}}'
    )


@pytest.mark.parametrize(
    "content, expected_context",
    [
        pytest.param(
            _feature_collection(_POLYGON_FEATURE),
            does_not_raise(),
            id="valid FeatureCollection with polygon",
        ),
        pytest.param(
            _feature_collection(),
            does_not_raise(),
            id="valid empty FeatureCollection",
        ),
        pytest.param(
            "not valid json {",
            pytest.raises(ValueError, match="not valid JSON"),
            id="invalid JSON",
        ),
        pytest.param(
            '{"type": "Feature", "geometry": null}',
            pytest.raises(
                ValueError,
                match="'features' is a required property",
            ),
            id="not a FeatureCollection",
        ),
        pytest.param(
            '{"type": "FeatureCollection"}',
            pytest.raises(
                ValueError,
                match="'features' is a required property",
            ),
            id="missing features key",
        ),
        pytest.param(
            _feature_collection('{"type": "Feature", "properties": {}}'),
            pytest.raises(
                ValueError,
                match="'geometry' is a required property",
            ),
            id="feature missing geometry",
        ),
        pytest.param(
            _feature_collection(
                '{"type": "Feature", "geometry": null, "properties": {}}'
            ),
            pytest.raises(
                ValueError,
                match="None is not of type 'object'",
            ),
            id="feature with null geometry",
        ),
        pytest.param(
            _feature_collection(
                '{"type": "Feature", '
                '"geometry": {"type": "Point", '
                '"coordinates": [0, 0]}, "properties": {}}'
            ),
            pytest.raises(
                ValueError,
                match="'Point' is not one of "
                "\\['Polygon', 'LineString', 'LinearRing'\\]",
            ),
            id="unsupported geometry type (Point)",
        ),
        pytest.param(
            _feature_collection(
                _feature_with_roi_type(
                    "LineString",
                    "[[0,0],[1,1]]",
                    "PolygonOfInterest",
                )
            ),
            pytest.raises(
                TypeError,
                match="does not match geometry type",
            ),
            id="roi_type mismatch: LineString/PolygonOfInterest",
        ),
        pytest.param(
            _feature_collection(
                _feature_with_roi_type(
                    "Polygon",
                    "[[[0,0],[1,0],[1,1],[0,0]]]",
                    "UnknownROI",
                )
            ),
            pytest.raises(
                ValueError,
                match="'UnknownROI' is not one of "
                "\\['PolygonOfInterest', 'LineOfInterest'\\]",
            ),
            id="unknown roi_type",
        ),
    ],
)
def test_roi_collection_geojson_validator(content, expected_context, tmp_path):
    """Test ValidROICollectionGeoJSON with valid and invalid inputs."""
    file_path = tmp_path / "test.geojson"
    file_path.write_text(content)
    with expected_context:
        validated = ValidROICollectionGeoJSON(file_path)
        assert validated.file == file_path
