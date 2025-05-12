"""Unit tests for the movement.io.utils module."""

import stat
from pathlib import Path

import pytest

from movement.io.utils import _validate_file_path
from movement.validators.files import ValidFile


@pytest.fixture
def sample_file_path():
    """Create a factory of file paths for a given suffix."""

    def _sample_file_path(tmp_path: Path, suffix: str):
        """Return a valid file path with the given suffix."""
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
    """Test file path validation with invalid permissions.

    S_IRUSR: Read permission for owner
    S_IRGRP: Read permission for group
    S_IROTH: Read permission for others
    """
    # Create a sample file with read-only permission
    file_path = sample_file_path(tmp_path, suffix)
    file_path.touch()
    file_path.chmod(
        stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH
    )  # Read-only permission (expected "write")

    # Try to validate the file path
    with pytest.raises(OSError):
        _validate_file_path(file_path, [suffix])


@pytest.mark.parametrize("suffix", [".txt", ".csv"])
def test_validate_file_path_file_exists(sample_file_path, tmp_path, suffix):
    """Test file path validation with a file that exists.

    S_IRUSR: Read permission for owner
    S_IWUSR: Write permission for owner
    S_IRGRP: Read permission for group
    S_IWGRP: Write permission for group
    S_IROTH: Read permission for others
    S_IWOTH: Write permission for others

    We include both read and write permissions because in real-world
    scenarios, it's very rare to have a file that is writable but not readable.
    """
    # Create a sample file with write permissions
    file_path = sample_file_path(tmp_path, suffix)
    file_path.touch()
    file_path.chmod(
        stat.S_IRUSR
        | stat.S_IWUSR
        | stat.S_IRGRP
        | stat.S_IWGRP
        | stat.S_IROTH
        | stat.S_IWOTH
    )  # Read-write permissions

    # Try to validate the file path
    with pytest.raises(OSError):
        _validate_file_path(file_path, [suffix])


@pytest.mark.parametrize("invalid_suffix", [".foo", "", None])
def test_validate_file_path_invalid_suffix(
    sample_file_path, tmp_path, invalid_suffix
):
    """Test file path validation with invalid file suffix."""
    # Create a valid txt file path
    file_path = sample_file_path(tmp_path, ".txt")

    # Try to validate using an invalid suffix
    with pytest.raises(ValueError):
        _validate_file_path(file_path, [invalid_suffix])


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
