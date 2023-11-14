import logging
import os
import stat

import h5py
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from movement.datasets import fetch_pose_data_path
from movement.io import PosesAccessor
from movement.logging import configure_logging


def pytest_configure():
    """Perform initial configuration for pytest.
    Fetches pose data file paths as a dictionary for tests."""
    pytest.POSE_DATA = {
        file_name: fetch_pose_data_path(file_name)
        for file_name in [
            "DLC_single-wasp.predictions.h5",
            "DLC_single-wasp.predictions.csv",
            "DLC_two-mice.predictions.csv",
            "SLEAP_single-mouse_EPM.analysis.h5",
            "SLEAP_single-mouse_EPM.predictions.slp",
            "SLEAP_three-mice_Aeon_proofread.analysis.h5",
            "SLEAP_three-mice_Aeon_proofread.predictions.slp",
            "SLEAP_three-mice_Aeon_mixed-labels.analysis.h5",
            "SLEAP_three-mice_Aeon_mixed-labels.predictions.slp",
        ]
    }


@pytest.fixture(autouse=True)
def setup_logging(tmp_path):
    """Set up logging for the test module.
    Redirects all logging to a temporary directory."""
    configure_logging(
        log_level=logging.DEBUG,
        logger_name="movement",
        log_directory=(tmp_path / ".movement"),
    )


@pytest.fixture
def unreadable_file(tmp_path):
    """Return a dictionary containing the file path and
    expected permission for an unreadable h5 file."""
    file_path = tmp_path / "unreadable.h5"
    with open(file_path, "w") as f:
        f.write("unreadable data")
    os.chmod(f.name, not stat.S_IRUSR)
    yield {
        "file_path": file_path,
        "expected_permission": "r",
    }
    os.chmod(f.name, stat.S_IRUSR)


@pytest.fixture
def unwriteable_file(tmp_path):
    """Return a dictionary containing the file path and
    expected permission for an unwriteable h5 file."""
    unwriteable_dir = tmp_path / "no_write"
    unwriteable_dir.mkdir()
    os.chmod(unwriteable_dir, not stat.S_IWUSR)
    file_path = unwriteable_dir / "unwriteable.h5"
    yield {
        "file_path": file_path,
        "expected_permission": "w",
    }
    os.chmod(unwriteable_dir, stat.S_IWUSR)


@pytest.fixture
def wrong_ext_file(tmp_path):
    """Return a dictionary containing the file path,
    expected permission, and expected suffix for a file
    with an incorrect extension.
    """
    file_path = tmp_path / "wrong_extension.txt"
    with open(file_path, "w") as f:
        f.write("")
    return {
        "file_path": file_path,
        "expected_permission": "r",
        "expected_suffix": ["h5", "csv"],
    }


@pytest.fixture
def nonexistent_file(tmp_path):
    """Return a dictionary containing the file path and
    expected permission for a nonexistent file."""
    file_path = tmp_path / "nonexistent.h5"
    return {
        "file_path": file_path,
        "expected_permission": "r",
    }


@pytest.fixture
def directory(tmp_path):  # used in save_poses, validators
    """Return a dictionary containing the file path and
    expected permission for a directory."""
    file_path = tmp_path / "directory"
    file_path.mkdir()
    return {
        "file_path": file_path,
        "expected_permission": "r",
    }


@pytest.fixture
def h5_file_no_dataframe(tmp_path):
    """Return a dictionary containing the file path and
    expected datasets for an h5 file with no dataframe."""
    file_path = tmp_path / "no_dataframe.h5"
    with h5py.File(file_path, "w") as f:
        f.create_dataset("data_in_list", data=[1, 2, 3])
    return {
        "file_path": file_path,
        "expected_datasets": ["dataframe"],
    }


@pytest.fixture
def fake_h5_file(tmp_path):  # used in save_poses, validators
    """Return a dictionary containing the file path,
    expected exception, and expected datasets for
    a file with .h5 extension that is not in HDF5 format.
    """
    file_path = tmp_path / "fake.h5"
    with open(file_path, "w") as f:
        f.write("")
    return {
        "file_path": file_path,
        "expected_datasets": ["dataframe"],
        "expected_permission": "w",
    }


@pytest.fixture
def invalid_single_animal_csv_file(tmp_path):
    """Return the file path for a fake single-animal csv file."""
    file_path = tmp_path / "fake_single_animal.csv"
    with open(file_path, "w") as f:
        f.write("scorer,columns\nsome,columns\ncoords,columns\n")
        f.write("1,2")
    return file_path


@pytest.fixture
def invalid_multi_animal_csv_file(tmp_path):
    """Return the file path for a fake multi-animal csv file."""
    file_path = tmp_path / "fake_multi_animal.csv"
    with open(file_path, "w") as f:
        f.write(
            "scorer,columns\nindividuals,columns\nbodyparts,columns\nsome,columns\n"
        )
        f.write("1,2")
    return file_path


@pytest.fixture
def dlc_style_df():
    """Return a valid DLC-style DataFrame."""
    return pd.read_hdf(pytest.POSE_DATA.get("DLC_single-wasp.predictions.h5"))


@pytest.fixture(
    params=[
        "SLEAP_single-mouse_EPM.analysis.h5",
        "SLEAP_single-mouse_EPM.predictions.slp",
        "SLEAP_three-mice_Aeon_proofread.analysis.h5",
        "SLEAP_three-mice_Aeon_proofread.predictions.slp",
        "SLEAP_three-mice_Aeon_mixed-labels.analysis.h5",
        "SLEAP_three-mice_Aeon_mixed-labels.predictions.slp",
    ]
)
def sleap_file(request):
    """Return the file path for a SLEAP .h5 or .slp file."""
    return pytest.POSE_DATA.get(request.param)


@pytest.fixture
def valid_tracks_array():
    """Return a function that generate different kinds
    of valid tracks array."""

    def _valid_tracks_array(array_type):
        """Return a valid tracks array."""
        if array_type == "single_keypoint_array":
            return np.zeros((10, 2, 1, 2))
        elif array_type == "single_track_array":
            return np.zeros((10, 1, 2, 2))
        else:  # "multi_track_array":
            return np.zeros((10, 2, 2, 2))

    return _valid_tracks_array


@pytest.fixture
def valid_pose_dataset(valid_tracks_array):
    """Return a valid pose tracks dataset."""
    dim_names = PosesAccessor.dim_names
    tracks_array = valid_tracks_array("multi_track_array")
    return xr.Dataset(
        data_vars={
            "pose_tracks": xr.DataArray(tracks_array, dims=dim_names),
            "confidence": xr.DataArray(
                tracks_array[..., 0],
                dims=dim_names[:-1],
            ),
        },
        coords={
            "time": np.arange(tracks_array.shape[0]),
            "individuals": ["ind1", "ind2"],
            "keypoints": ["key1", "key2"],
            "space": ["x", "y"],
        },
        attrs={
            "fps": None,
            "time_unit": "frames",
            "source_software": "SLEAP",
            "source_file": "test.h5",
        },
    )
