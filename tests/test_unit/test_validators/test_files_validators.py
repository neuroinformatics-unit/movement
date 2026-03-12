import re
from contextlib import nullcontext as does_not_raise
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pandas as pd
import pytest
from attrs import define, field

from movement.validators.files import (
    DEFAULT_FRAME_REGEXP,
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
    "frame_regexp, log_message",
    [
        (
            r"\d+",
            "The regexp pattern must contain exactly one capture "
            r"group for the frame number (got \d+).",
        ),  # no capture group
        (
            r"(\d+)\.(\w+)",
            "The regexp pattern must contain exactly one capture "
            r"group for the frame number (got (\d+)\.(\w+)).",
        ),  # two capture groups
        (
            r"*",
            "regular expression for the frame numbers (*) "
            "could not be compiled.",
        ),  # compilation error
    ],
)
def test_via_tracks_validator_invalid_frame_regexp(frame_regexp, log_message):
    """Test _frame_regexp_valid rejects invalid patterns."""
    with pytest.raises(ValueError, match=re.escape(log_message)):
        ValidVIATracksCSV._frame_regexp_valid(None, frame_regexp)


def test_via_tracks_validator_invalid_header(via_tracks_csv_factory):
    """Test _file_contains_valid_header rejects a wrong header."""
    invalid_df = pd.read_csv(via_tracks_csv_factory("via_invalid_header"))
    with pytest.raises(
        ValueError, match=re.escape(".csv header row does not match")
    ):
        ValidVIATracksCSV._file_contains_valid_header(None, invalid_df)


@pytest.mark.parametrize(
    "invalid_input, log_message",
    [
        (
            "via_frame_number_in_file_attribute_not_integer",
            "Some frame numbers cannot be cast as integer. ",
        ),
        (
            "via_frame_number_in_filename_wrong_pattern",
            "Could not extract frame numbers from the filenames "
            r"using the regular expression (0\d*)\.\w+$.",
        ),
        (
            "via_more_frame_numbers_than_filenames",
            "number of unique frame numbers does not match",
        ),
        (
            "via_less_frame_numbers_than_filenames",
            "number of unique frame numbers does not match",
        ),
    ],
)
def test_via_tracks_validator_invalid_frame_numbers(
    via_tracks_csv_factory, invalid_input, log_message
):
    """Test _file_contains_valid_frame_numbers rejects bad frames."""
    invalid_df = pd.read_csv(via_tracks_csv_factory(invalid_input))

    # mock "self" with default frame regexp attribute
    mock_self = MagicMock()
    mock_self.frame_regexp = DEFAULT_FRAME_REGEXP
    with pytest.raises(ValueError, match=re.escape(log_message)):
        ValidVIATracksCSV._file_contains_valid_frame_numbers(
            mock_self, invalid_df
        )


@pytest.mark.parametrize(
    "invalid_input, log_message",
    [
        (
            "via_region_shape_attribute_not_rect",
            "The bounding box in row 1 shape was expected to be "
            "'rect' (rectangular) but instead got circle.",
        ),
        (
            "via_region_shape_attribute_missing_x",
            "The bounding box in row 1 is missing "
            "a geometric parameter (x, y, width, height). ",
        ),
        (
            "via_region_attribute_missing_track",
            "The bounding box in row 1 is missing a track ID. ",
        ),
        (
            "via_track_id_not_castable_as_int",
            "The track ID of the bounding box in row 1 cannot be "
            "cast as an integer (got track ID 'FOO').",
        ),
    ],
)
def test_via_tracks_validator_invalid_tracked_bboxes(
    via_tracks_csv_factory, invalid_input, log_message
):
    """Test _file_contains_tracked_bboxes rejects bad bbox data."""
    invalid_df = pd.read_csv(via_tracks_csv_factory(invalid_input))
    with pytest.raises(ValueError, match=re.escape(log_message)):
        ValidVIATracksCSV._file_contains_tracked_bboxes(None, invalid_df)


def test_via_tracks_validator_duplicate_track_ids_per_frame(
    via_tracks_csv_factory,
):
    """Test _file_contains_unique_track_ids_per_filename rejects
    duplicate IDs within a frame.
    """
    invalid_df = pd.read_csv(
        via_tracks_csv_factory("via_track_ids_not_unique_per_frame")
    )
    _, _, _, _, ids, _ = ValidVIATracksCSV._file_contains_tracked_bboxes(
        None, invalid_df
    )
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Duplicate track IDs found in the following files: "
            "['04.09.2023-04-Right_RE_test_frame_01.png']. "
        ),
    ):
        ValidVIATracksCSV._file_contains_unique_track_ids_per_filename(
            None, invalid_df, ids
        )


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
