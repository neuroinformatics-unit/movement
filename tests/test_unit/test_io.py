import os
from pathlib import Path

import h5py
import pandas as pd
import pytest
from pydantic import ValidationError

from movement.io import load_poses


@pytest.fixture
def valid_dlc_h5_files(tmp_path):
    """Return the paths to valid DLC poses files,
    in .h5 format.

    Returns
    -------
    dict
        Dictionary containing the paths.
        - h5_path: pathlib Path to a valid .h5 file
        - h5_str: path as str to a valid .h5 file
    """
    test_data_dir = Path(__file__).parent.parent.parent / "data"
    h5_file = test_data_dir / "DLC_sample_poses.h5"
    return {
        "h5_path": h5_file,
        "h5_str": h5_file.as_posix(),
    }


@pytest.fixture
def invalid_files(tmp_path):
    unreadable_file = tmp_path / "unreadable.h5"
    with open(unreadable_file, "w") as f:
        f.write("unreadable data")
        os.chmod(f.name, 0o000)

    wrong_ext_file = tmp_path / "wrong_extension.txt"
    with open(wrong_ext_file, "w") as f:
        f.write("")

    h5_file_no_dataframe = tmp_path / "no_dataframe.h5"
    with h5py.File(h5_file_no_dataframe, "w") as f:
        f.create_dataset("df_with_missing", data=[1, 2, 3])

    h5_file_empty_dataframe = tmp_path / "empty_dataframe.h5"
    with h5py.File(h5_file_empty_dataframe, "w") as f:
        f.create_dataset("df_with_missing", data=pd.DataFrame())

    nonexistent_file = tmp_path / "nonexistent.h5"

    return {
        "unreadable": unreadable_file,
        "wrong_ext": wrong_ext_file,
        "no_dataframe": h5_file_no_dataframe,
        "empty_dataframe": h5_file_empty_dataframe,
        "nonexistent": nonexistent_file,
    }


def test_load_valid_dlc_h5_files(valid_dlc_h5_files):
    """Test loading valid DLC poses files."""
    for file_type, file_path in valid_dlc_h5_files.items():
        poses = load_poses.from_dlc_h5(file_path)
        assert isinstance(poses, pd.DataFrame)


def test_load_invalid_dlc_h5_files(invalid_files):
    """Test loading invalid DLC poses files."""
    for file_type, file_path in invalid_files.items():
        if file_type == "nonexistent":
            with pytest.raises(FileNotFoundError):
                load_poses.from_dlc_h5(file_path)
        elif file_type == "wrong_ext":
            with pytest.raises(ValueError):
                load_poses.from_dlc_h5(file_path)
        else:
            with pytest.raises(OSError):
                load_poses.from_dlc_h5(file_path)


@pytest.mark.parametrize("filepath", [1, 1.0, True, None, [], {}])
def test_load_from_dlc_with_incorrect_filepath_types(filepath):
    """Test loading poses from a file with an incorrect type."""
    with pytest.raises(ValidationError):
        load_poses.from_dlc_h5(filepath)
