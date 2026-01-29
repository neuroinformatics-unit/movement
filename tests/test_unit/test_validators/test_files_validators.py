import stat
from pathlib import Path

import pytest

from movement.validators.files import (
    ValidAniposeCSV,
    ValidDeepLabCutCSV,
    ValidFile,
    ValidHDF5,
    ValidNWBFile,
    ValidVIATracksCSV,
    _validate_file_path,
)


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
