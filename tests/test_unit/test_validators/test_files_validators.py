import pytest

from movement.validators.files import (
    ValidDeepLabCutCSV,
    ValidFile,
    ValidHDF5,
    ValidVIAtracksCSV,
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


@pytest.mark.skip("Fixtures not implemented yet")
@pytest.mark.parametrize(
    "invalid_input, log_message",
    [
        (
            "invalid_header",
            ".csv header row does not match the known format for "
            "VIA tracks output files.",
        ),
        (
            "frame_number_as_attribute_not_integer",
            "'frame' attribute for file 'bla.csv' "
            "cannot be cast as an integer. Please review the "
            "file's attributes: '\{bla\:\}'.",
        ),
        (
            "frame_number_as_filename_wrong_pattern",
            "A frame number could not be extracted for filename "
            "bla.png. Please review the VIA tracks csv file. If the "
            "frame number is included in the filename, it is "
            "expected as a zero-padded integer between an "
            "underscore '_' and the file extension "
            "(e.g. img_00234.png).",
        ),
        (
            "frame_numbers_do_not_match_filenames",
            "The number of unique frame numbers does not match the number "
            "of unique files. Please review the VIA tracks csv file and "
            "ensure a unique frame number is defined for each filename. "
            "This can by done via a 'frame' file attribute, or by "
            "including the frame number in the filename. If included in "
            "the filename, the frame number is expected as a zero-padded "
            "integer between an underscore '_' and the file extension "
            "(e.g. img_00234.png).",
        ),
        (
            "region_shape_attributes_name_not_rect",
            "Bounding box shape must be 'rect' but instead got "
            "'patata' for file 'bla.csv' (row 0, 0-based).",
        ),
        (
            "region_shape_attributes_missing_x",
            "At least one bounding box shape parameter is missing. "
            "Expected 'x', 'y', 'width', 'height' to exist as "
            "'region_shape_attributes', but got "
            "'\{bla\:blah\}' for file bla.csv (row 0, 0-based).,",
        ),
        (
            "region_shape_attributes_missing_y",
            "At least one bounding box shape parameter is missing. "
            "Expected 'x', 'y', 'width', 'height' to exist as "
            "'region_shape_attributes', but got "
            "'\{bla\:blah\}' for file bla.csv (row 0, 0-based).,",
        ),
        (
            "region_shape_attributes_missing_width",
            "At least one bounding box shape parameter is missing. "
            "Expected 'x', 'y', 'width', 'height' to exist as "
            "'region_shape_attributes', but got "
            "'\{bla\:blah\}' for file bla.csv (row 0, 0-based).,",
        ),
        (
            "region_shape_attributes_missing_height",
            "At least one bounding box shape parameter is missing. "
            "Expected 'x', 'y', 'width', 'height' to exist as "
            "'region_shape_attributes', but got "
            "'\{bla\:blah\}' for file bla.csv (row 0, 0-based).,",
        ),
        (
            "region_attributes_missing_track",
            "Bounding box in file bla.csv and row 0 "
            "(0-based) does not have a 'track' attribute defined. "
            "Please review the VIA tracks csv file and ensure that "
            "all bounding boxes have a 'track' field under "
            "'region_attributes'.",
        ),
        (
            "track_id_not_castable_as_int",
            "The track ID for the bounding box in file "
            "bla.csv and row 0 is 'patata', which "
            "cannot be cast as an integer. "
            "Please review the VIA tracks csv file.",
        ),
        (
            "track_ids_not_unique_per_frame",
            "Multiple bounding boxes have the same track ID "
            "in file bla.csv. Please review the VIA tracks csv file.",
        ),
    ],
)
def test_via_tracks_csv_validator_with_invalid_input(
    invalid_input, log_message, request
):
    """Test that invalid VIA tracks CSV files raise the appropriate errors.

    Errors to check:
    - error if csv header is wrong
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
        ValidVIAtracksCSV(file_path)

    assert str(excinfo.value) == log_message
