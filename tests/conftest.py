"""Fixtures and configurations applied to the entire test suite."""

import logging
import os
from pathlib import Path
from unittest.mock import mock_open, patch

import h5py
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from movement import MovementDataset
from movement.sample_data import fetch_dataset_paths, list_datasets
from movement.utils.logging import configure_logging


def pytest_configure():
    """Perform initial configuration for pytest.
    Fetches pose data file paths as a dictionary for tests.
    """
    pytest.DATA_PATHS = {}
    for file_name in list_datasets():
        paths_dict = fetch_dataset_paths(file_name)
        data_path = paths_dict.get("poses") or paths_dict.get("bboxes")
        pytest.DATA_PATHS[file_name] = data_path


@pytest.fixture(autouse=True)
def setup_logging(tmp_path):
    """Set up logging for the test module.
    Redirects all logging to a temporary directory.
    """
    configure_logging(
        log_level=logging.DEBUG,
        logger_name="movement",
        log_directory=(tmp_path / ".movement"),
    )


@pytest.fixture
def unreadable_file(tmp_path):
    """Return a dictionary containing the file path and
    expected permission for an unreadable .h5 file.
    """
    file_path = tmp_path / "unreadable.h5"
    file_mock = mock_open()
    file_mock.return_value.read.side_effect = PermissionError
    with (
        patch("builtins.open", side_effect=file_mock),
        patch.object(Path, "exists", return_value=True),
    ):
        yield {
            "file_path": file_path,
            "expected_permission": "r",
        }


@pytest.fixture
def unwriteable_file(tmp_path):
    """Return a dictionary containing the file path and
    expected permission for an unwriteable .h5 file.
    """
    unwriteable_dir = tmp_path / "no_write"
    unwriteable_dir.mkdir()
    original_access = os.access

    def mock_access(path, mode):
        if path == unwriteable_dir and mode == os.W_OK:
            return False
        # Ensure that the original access function is called
        # for all other cases
        return original_access(path, mode)

    with patch("os.access", side_effect=mock_access):
        file_path = unwriteable_dir / "unwriteable.h5"
        yield {
            "file_path": file_path,
            "expected_permission": "w",
        }


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
    expected permission for a nonexistent file.
    """
    file_path = tmp_path / "nonexistent.h5"
    return {
        "file_path": file_path,
        "expected_permission": "r",
    }


@pytest.fixture
def directory(tmp_path):
    """Return a dictionary containing the file path and
    expected permission for a directory.
    """
    file_path = tmp_path / "directory"
    file_path.mkdir()
    return {
        "file_path": file_path,
        "expected_permission": "r",
    }


@pytest.fixture
def h5_file_no_dataframe(tmp_path):
    """Return a dictionary containing the file path and
    expected datasets for a .h5 file with no dataframe.
    """
    file_path = tmp_path / "no_dataframe.h5"
    with h5py.File(file_path, "w") as f:
        f.create_dataset("data_in_list", data=[1, 2, 3])
    return {
        "file_path": file_path,
        "expected_datasets": ["dataframe"],
    }


@pytest.fixture
def fake_h5_file(tmp_path):
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
def invalid_single_individual_csv_file(tmp_path):
    """Return the file path for a fake single-individual .csv file."""
    file_path = tmp_path / "fake_single_individual.csv"
    with open(file_path, "w") as f:
        f.write("scorer,columns\nsome,columns\ncoords,columns\n")
        f.write("1,2")
    return file_path


@pytest.fixture
def invalid_multi_individual_csv_file(tmp_path):
    """Return the file path for a fake multi-individual .csv file."""
    file_path = tmp_path / "fake_multi_individual.csv"
    with open(file_path, "w") as f:
        f.write(
            "scorer,columns\nindividuals,columns\nbodyparts,columns\nsome,columns\n"
        )
        f.write("1,2")
    return file_path


@pytest.fixture
def new_file_wrong_ext(tmp_path):
    """Return the file path for a new file with the wrong extension."""
    return tmp_path / "new_file_wrong_ext.txt"


@pytest.fixture
def new_h5_file(tmp_path):
    """Return the file path for a new .h5 file."""
    return tmp_path / "new_file.h5"


@pytest.fixture
def new_csv_file(tmp_path):
    """Return the file path for a new .csv file."""
    return tmp_path / "new_file.csv"


@pytest.fixture
def dlc_style_df():
    """Return a valid DLC-style DataFrame."""
    return pd.read_hdf(pytest.DATA_PATHS.get("DLC_single-wasp.predictions.h5"))


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
    return pytest.DATA_PATHS.get(request.param)


@pytest.fixture
def valid_position_array():
    """Return a function that generates different kinds
    of a valid position array.
    """

    def _valid_position_array(array_type):
        """Return a valid position array."""
        # Unless specified, default is a multi_individual_array with
        # 10 frames, 2 individuals, and 2 keypoints.
        n_frames = 10
        n_individuals = 2
        n_keypoints = 2
        base = np.arange(n_frames, dtype=float)[
            :, np.newaxis, np.newaxis, np.newaxis
        ]
        if array_type == "single_keypoint_array":
            n_keypoints = 1
        elif array_type == "single_individual_array":
            n_individuals = 1
        x_points = np.repeat(base * base, n_individuals * n_keypoints)
        y_points = np.repeat(base * 4, n_individuals * n_keypoints)
        position_array = np.ravel(np.column_stack((x_points, y_points)))
        return position_array.reshape(n_frames, n_individuals, n_keypoints, 2)

    return _valid_position_array


@pytest.fixture
def valid_poses_dataset(valid_position_array, request):
    """Return a valid pose tracks dataset."""
    dim_names = MovementDataset.dim_names
    # create a multi_individual_array by default unless overridden via param
    try:
        array_format = request.param
    except AttributeError:
        array_format = "multi_individual_array"
    position_array = valid_position_array(array_format)
    n_frames, n_individuals, n_keypoints = position_array.shape[:3]
    return xr.Dataset(
        data_vars={
            "position": xr.DataArray(position_array, dims=dim_names),
            "confidence": xr.DataArray(
                np.repeat(
                    np.linspace(0.1, 1.0, n_frames),
                    n_individuals * n_keypoints,
                ).reshape(position_array.shape[:-1]),
                dims=dim_names[:-1],
            ),
        },
        coords={
            "time": np.arange(n_frames),
            "individuals": [f"ind{i}" for i in range(1, n_individuals + 1)],
            "keypoints": [f"key{i}" for i in range(1, n_keypoints + 1)],
            "space": ["x", "y"],
        },
        attrs={
            "fps": None,
            "time_unit": "frames",
            "source_software": "SLEAP",
            "source_file": "test.h5",
        },
    )


@pytest.fixture
def valid_poses_dataset_with_nan(valid_poses_dataset):
    """Return a valid pose tracks dataset with NaN values."""
    valid_poses_dataset.position.loc[
        {"individuals": "ind1", "time": [3, 7, 8]}
    ] = np.nan
    return valid_poses_dataset


@pytest.fixture
def not_a_dataset():
    """Return data that is not a pose tracks dataset."""
    return [1, 2, 3]


@pytest.fixture
def empty_dataset():
    """Return an empty pose tracks dataset."""
    return xr.Dataset()


@pytest.fixture
def missing_var_dataset(valid_poses_dataset):
    """Return a pose tracks dataset missing an expected variable."""
    return valid_poses_dataset.drop_vars("position")


@pytest.fixture
def missing_dim_dataset(valid_poses_dataset):
    """Return a pose tracks dataset missing an expected dimension."""
    return valid_poses_dataset.rename({"time": "tame"})


@pytest.fixture(
    params=[
        "not_a_dataset",
        "empty_dataset",
        "missing_var_dataset",
        "missing_dim_dataset",
    ]
)
def invalid_poses_dataset(request):
    """Return an invalid pose tracks dataset."""
    return request.getfixturevalue(request.param)


@pytest.fixture(params=["displacement", "velocity", "acceleration"])
def kinematic_property(request):
    """Return a kinematic property."""
    return request.param


@pytest.fixture
def invalid_header_via_tracks_csv(tmp_path):
    """Return the file path for an invalid VIA tracks csv file."""
    file_path = tmp_path / "invalid_header_via_tracks_csv.csv"
    with open(file_path, "w") as f:
        f.write("filename,file_size,file_attributes\n")
        f.write("1,2,3")
    return file_path


class Helpers:
    """Generic helper methods for ``movement`` test modules."""

    @staticmethod
    def count_nans(da):
        """Count number of NaNs in a DataArray."""
        return da.isnull().sum().item()

    @staticmethod
    def count_consecutive_nans(da):
        """Count occurrences of consecutive NaNs in a DataArray."""
        return (da.isnull().astype(int).diff("time") == 1).sum().item()


@pytest.fixture
def helpers():
    """Return an instance of the ``Helpers`` class."""
    return Helpers
