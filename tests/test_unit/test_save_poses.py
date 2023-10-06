from contextlib import nullcontext as does_not_raise

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pytest import POSE_DATA

from movement.io import PosesAccessor, load_poses, save_poses


class TestSavePoses:
    """Test suite for the save_poses module."""

    @pytest.fixture
    def valid_pose_dataset(self, valid_tracks_array):
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

    @pytest.fixture
    def not_a_dataset(self):
        """Return an invalid pose tracks dataset."""
        return [1, 2, 3]

    @pytest.fixture
    def empty_dataset(self):
        """Return an empty pose tracks dataset."""
        return xr.Dataset()

    @pytest.fixture
    def missing_var_dataset(self, valid_pose_dataset):
        """Return a pose tracks dataset missing a variable."""
        return valid_pose_dataset.drop_vars("pose_tracks")

    @pytest.fixture
    def missing_dim_dataset(self, valid_pose_dataset):
        """Return a pose tracks dataset missing a dimension."""
        return valid_pose_dataset.drop_dims("time")

    @pytest.fixture(
        params=[
            "not_a_dataset",
            "empty_dataset",
            "missing_var_dataset",
            "missing_dim_dataset",
        ]
    )
    def invalid_pose_datasets(self, request):
        """Return a list of invalid pose tracks datasets."""
        return request.param

    @pytest.mark.parametrize(
        "ds, expected_exception",
        [
            (np.array([1, 2, 3]), pytest.raises(ValueError)),  # incorrect type
            (
                load_poses.from_dlc_file(
                    POSE_DATA.get("DLC_single-wasp.predictions.h5")
                ),
                does_not_raise(),
            ),  # valid dataset
            (
                load_poses.from_dlc_file(
                    POSE_DATA.get("DLC_two-mice.predictions.csv")
                ),
                does_not_raise(),
            ),  # valid dataset
        ],
    )
    def test_to_dlc_df(self, ds, expected_exception):
        """Test that converting a valid/invalid xarray dataset to
        a DeepLabCut-style pandas DataFrame returns the expected result."""
        with expected_exception as e:
            df = save_poses.to_dlc_df(ds)
            if e is None:  # valid input
                assert isinstance(df, pd.DataFrame)
                assert isinstance(df.columns, pd.MultiIndex)
                assert df.columns.names == [
                    "scorer",
                    "individuals",
                    "bodyparts",
                    "coords",
                ]

    @pytest.fixture
    def new_dlc_h5_file(self, tmp_path):
        """Return a dictionary containing path, expected exception,
        and expected permission for a new DLC h5 file."""
        file_path = tmp_path / "new_dlc.h5"
        return {"file_path": file_path}

    @pytest.fixture
    def new_dlc_csv_file(self, tmp_path):
        """Return a dictionary containing path, expected exception,
        and expected permission for a new DLC csv file."""
        file_path = tmp_path / "new_dlc.csv"
        return {"file_path": file_path}

    @pytest.mark.parametrize(
        "input, expected_exception",
        [
            (
                "fake_h5_file",
                pytest.raises(FileExistsError),
            ),  # invalid file path
            (
                "new_file_wrong_ext",
                pytest.raises(ValueError),
            ),  # invalid file path
            (
                "directory",
                pytest.raises(IsADirectoryError),
            ),  # invalid file path
            ("new_dlc_h5_file", does_not_raise()),  # valid file path
            ("new_dlc_csv_file", does_not_raise()),  # valid file path
        ],
    )
    def test_to_dlc_file_valid_dataset(
        self, input, expected_exception, valid_pose_dataset, request
    ):
        """Test that saving a valid pose dataset to a valid/invalid
        DeepLabCut-style file returns the appropriate errors."""
        with expected_exception:
            file_path = request.getfixturevalue(input).get("file_path")
            save_poses.to_dlc_file(valid_pose_dataset, file_path)

    def test_to_dlc_file_invalid_dataset(
        self, invalid_pose_datasets, request, tmp_path
    ):
        """Test that saving an invalid pose dataset to a valid
        DeepLabCut-style file returns the appropriate errors."""
        with pytest.raises(ValueError):
            save_poses.to_dlc_file(
                request.getfixturevalue(invalid_pose_datasets),
                tmp_path / "test.h5",
            )
