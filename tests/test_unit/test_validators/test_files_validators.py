from contextlib import nullcontext as does_not_raise
from pathlib import Path
from unittest.mock import Mock

import pytest
from attrs import define, field

from movement.validators.files import (
    ValidAniposeCSV,
    ValidDeepLabCutCSV,
    ValidNWBFile,
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
        (123, pytest.raises(TypeError)),
        (None, pytest.raises(TypeError)),
    ],
)
def test_nwb_file_validator_with_invalid_input(
    invalid_input, expected_exception
):
    """Test that invalid NWB file inputs raise the appropriate errors."""
    with expected_exception:
        ValidNWBFile(invalid_input)
