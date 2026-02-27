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
    "invalid_input, error_type, log_message",
    [
        (
            "via_invalid_header",
            ValueError,
            ".csv header row does not match",
        ),
        (
            "via_frame_number_in_file_attribute_not_integer",
            ValueError,
            "Extracted frame number 'FOO' cannot be cast as integer. ",
        ),
        (
            "via_frame_number_in_filename_wrong_pattern",
            ValueError,
            "Could not extract frame numbers from the filenames using "
            r"the regular expression (0\d*)\.\w+$. Please ensure "
            "filenames match the expected pattern, or define the "
            "frame numbers in file_attributes.",
        ),
        (
            "via_more_frame_numbers_than_filenames",
            ValueError,
            "number of unique frame numbers does not match",
        ),
        (
            "via_less_frame_numbers_than_filenames",
            ValueError,
            "number of unique frame numbers does not match",
        ),
        (
            "via_region_shape_attribute_not_rect",
            ValueError,
            "The bounding box in row 1 shape was expected to be 'rect' "
            "(rectangular) but instead got circle.",
        ),
        (
            "via_region_shape_attribute_missing_x",
            ValueError,
            "The bounding box in row 1 is missing "
            "a geometric parameter (x, y, width, height). ",
        ),
        (
            "via_region_attribute_missing_track",
            ValueError,
            "The bounding box in row 1 is missing a track ID. ",
        ),
        (
            "via_track_id_not_castable_as_int",
            ValueError,
            "The track ID of the bounding box in row 1 cannot be "
            "cast as an integer (got track ID 'FOO').",
        ),
        (
            "via_track_ids_not_unique_per_frame",
            ValueError,
            "Duplicate track IDs found in the following files: "
            "['04.09.2023-04-Right_RE_test_frame_01.png']. ",
        ),
    ],
)
def test_via_tracks_csv_validator_with_invalid_input(
    via_tracks_csv_factory, invalid_input, error_type, log_message
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
    file_path = via_tracks_csv_factory(invalid_input)
    with pytest.raises(error_type) as excinfo:
        ValidVIATracksCSV(file_path)

    assert log_message in str(excinfo.value)


@pytest.mark.parametrize(
    "invalid_regexp",
    [
        r"\d+",  # no capture group
        r"(\d+)\.(\w+)",  # two capture groups
    ],
)
def test_via_tracks_csv_validator_with_invalid_regexp(
    via_tracks_csv_factory, invalid_regexp
):
    """Test regexp with wrong number of capture groups raises ValueError."""
    file_path = via_tracks_csv_factory("via_valid")
    with pytest.raises(ValueError) as excinfo:
        ValidVIATracksCSV(file_path, frame_regexp=invalid_regexp)

    assert (
        "The regexp pattern must contain exactly one capture group for the "
        rf"frame number (got {invalid_regexp})."
    ) in str(excinfo.value)


def test_via_tracks_csv_validator_attributes(
    via_tracks_csv_factory,
):
    """Test that the attributes are as expected after validation.

    The dataframe attribute should be cleared after validation.
    The pre-parsed data should be defined.
    """
    file_path = via_tracks_csv_factory("via_valid")
    validator = ValidVIATracksCSV(file_path)

    # Check that the dataframe attribute is cleared
    assert validator.df is None

    # Check that the pre-parsed data is defined
    assert isinstance(validator.x, list)
    assert isinstance(validator.y, list)
    assert isinstance(validator.w, list)
    assert isinstance(validator.h, list)
    assert isinstance(validator.ids, list)
    assert isinstance(validator.frame_numbers, list)
    assert isinstance(validator.confidence_values, list)


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
