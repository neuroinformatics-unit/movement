"""Unit tests for the movement.io.utils module."""

import stat
from pathlib import Path

import pytest

from movement.validators.files import ValidFile, _validate_file_path


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
