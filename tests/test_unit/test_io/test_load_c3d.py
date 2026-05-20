"""Test the C3D loader functionality."""

import pytest

from movement.io.load_c3d import from_c3d_file


def test_load_c3d_file_not_found():
    """Test that the loader gracefully catches missing files."""
    fake_path = "this_file_does_not_exist.c3d"
    with pytest.raises(FileNotFoundError, match="C3D file not found"):
        from_c3d_file(fake_path)


def test_load_c3d_invalid_file(tmp_path):
    """Test that the loader gracefully catches corrupted C3D files."""
    broken_file = tmp_path / "broken.c3d"
    broken_file.write_text("This is definitely not a real C3D file.")
    with pytest.raises(ValueError, match="Failed to load C3D file"):
        from_c3d_file(broken_file)
