import pytest

from movement.io.validators.files import (
    ValidDeepLabCutCSV,
    ValidFile,
    ValidHDF5,
)


class TestValidators:
    """Test suite for the validators module."""

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
        self, invalid_input, expected_exception, request
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
        self, invalid_input, expected_exception, request
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
        self, invalid_input, expected_exception, request
    ):
        """Test that invalid CSV files raise the appropriate errors."""
        file_path = request.getfixturevalue(invalid_input)
        with expected_exception:
            ValidDeepLabCutCSV(file_path)
