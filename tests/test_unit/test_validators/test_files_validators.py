import pytest

from movement.validators.files import (
    ValidDeepLabCutCSV,
    ValidFile,
    ValidHDF5,
    ValidVIATracksCSV,
)


@pytest.mark.parametrize(
    "invalid_input, expected_exception",
    [
        ("unreadable_file", pytest.raises(PermissionError)),
        ("unwriteable_file", pytest.raises(PermissionError)),
        ("fake_h5_file", pytest.raises(FileExistsError)),
        ("wrong_ext_file", pytest.raises(ValueError)),
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
        ("h5_file_no_dataframe", pytest.raises(ValueError)),
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
    "invalid_input, log_message",
    [
        (
            "via_tracks_csv_with_invalid_header",
            ".csv header row does not match the known format for "
            "VIA tracks output files. "
            "Expected "
            "['filename', 'file_size', 'file_attributes', "
            "'region_count', 'region_id', 'region_shape_attributes', "
            "'region_attributes'] "
            "but got ['filename', 'file_size', 'file_attributes'].",
        ),
        (
            "frame_number_in_file_attribute_not_integer",
            "04.09.2023-04-Right_RE_test_frame_A.png (row 0): "
            "'frame' file attribute cannot be cast as an integer. "
            "Please review the file attributes: "
            "{'clip': 123, 'frame': 'FOO'}.",
        ),
        (
            "frame_number_in_filename_wrong_pattern",
            "04.09.2023-04-Right_RE_test_frame_1.png (row 0): "
            "a frame number could not be extracted from the filename. "
            "If included in the filename, the frame number is "
            "expected as a zero-padded integer between an "
            "underscore '_' and the file extension "
            "(e.g. img_00234.png).",
        ),
        (
            "more_frame_numbers_than_filenames",
            "The number of unique frame numbers does not match the number "
            "of unique image files. Please review the VIA tracks .csv file "
            "and ensure a unique frame number is defined for each file. ",
        ),
        (
            "less_frame_numbers_than_filenames",
            "The number of unique frame numbers does not match the number "
            "of unique image files. Please review the VIA tracks .csv file "
            "and ensure a unique frame number is defined for each file. ",
        ),
        (
            "region_shape_attribute_not_rect",
            "04.09.2023-04-Right_RE_test_frame_01.png (row 0): "
            "bounding box shape must be 'rect' (rectangular) "
            "but instead got 'circle'.",
        ),
        (
            "region_shape_attribute_missing_x",
            "04.09.2023-04-Right_RE_test_frame_01.png (row 0): "
            "at least one bounding box shape parameter is missing. "
            "Expected 'x', 'y', 'width', 'height' to exist as "
            "'region_shape_attributes', but got "
            "'['name', 'y', 'width', 'height']'.",
        ),
        (
            "region_attribute_missing_track",
            "04.09.2023-04-Right_RE_test_frame_01.png (row 0): "
            "bounding box does not have a 'track' attribute defined "
            "under 'region_attributes'. "
            "Please review the VIA tracks .csv file.",
        ),
        (
            "track_id_not_castable_as_int",
            "04.09.2023-04-Right_RE_test_frame_01.png (row 0): "
            "the track ID for the bounding box cannot be cast "
            "as an integer. "
            "Please review the VIA tracks .csv file.",
        ),
        (
            "track_ids_not_unique_per_frame",
            "04.09.2023-04-Right_RE_test_frame_01.png: "
            "multiple bounding boxes in this file have the same track ID. "
            "Please review the VIA tracks .csv file.",
        ),
    ],
)
def test_via_tracks_csv_validator_with_invalid_input(
    invalid_input, log_message, request
):
    """Test that invalid VIA tracks CSV files raise the appropriate errors.

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
    file_path = request.getfixturevalue(invalid_input)
    with pytest.raises(ValueError) as excinfo:
        ValidVIATracksCSV(file_path)

    assert str(excinfo.value) == log_message
