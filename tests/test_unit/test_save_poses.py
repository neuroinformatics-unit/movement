from contextlib import nullcontext as does_not_raise

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pytest import POSE_DATA

from movement.io import load_poses, save_poses


class TestSavePoses:
    """Test suite for the save_poses module."""

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
    def new_file_wrong_ext(self, tmp_path):
        """Return the file path for a new file with the wrong extension."""
        return tmp_path / "new_file_wrong_ext.txt"

    @pytest.fixture
    def new_dlc_h5_file(self, tmp_path):
        """Return the file path for a new DeepLabCut .h5 file."""
        return tmp_path / "new_dlc_file.h5"

    @pytest.fixture
    def new_dlc_csv_file(self, tmp_path):
        """Return the file path for a new DeepLabCut .csv file."""
        return tmp_path / "new_dlc_file.csv"

    @pytest.fixture
    def missing_dim_dataset(self, valid_pose_dataset):
        """Return a pose tracks dataset missing a dimension."""
        return valid_pose_dataset.drop_dims("time")

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
            (
                load_poses.from_sleap_file(
                    POSE_DATA.get("SLEAP_single-mouse_EPM.analysis.h5")
                ),
                does_not_raise(),
            ),  # valid dataset
            (
                load_poses.from_sleap_file(
                    POSE_DATA.get(
                        "SLEAP_three-mice_Aeon_proofread.predictions.slp"
                    )
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

    @pytest.mark.parametrize(
        "file_fixture, expected_exception",
        [
            (
                "fake_h5_file",
                pytest.raises(FileExistsError),
            ),  # invalid file path
            (
                "directory",
                pytest.raises(IsADirectoryError),
            ),  # invalid file path
            (
                "new_file_wrong_ext",
                pytest.raises(ValueError),
            ),  # invalid file path
            ("new_dlc_h5_file", does_not_raise()),  # valid file path
            ("new_dlc_csv_file", does_not_raise()),  # valid file path
        ],
    )
    def test_to_dlc_file_valid_dataset(
        self, file_fixture, expected_exception, valid_pose_dataset, request
    ):
        """Test that saving a valid pose dataset to a valid/invalid
        DeepLabCut-style file returns the appropriate errors."""
        with expected_exception:
            val = request.getfixturevalue(file_fixture)
            file_path = val.get("file_path") if isinstance(val, dict) else val
            save_poses.to_dlc_file(valid_pose_dataset, file_path)

    @pytest.mark.parametrize(
        "invalid_pose_dataset",
        [
            "not_a_dataset",
            "empty_dataset",
            "missing_var_dataset",
            "missing_dim_dataset",
        ],
    )
    def test_to_dlc_file_invalid_dataset(
        self, invalid_pose_dataset, request, tmp_path
    ):
        """Test that saving an invalid pose dataset to a valid
        DeepLabCut-style file returns the appropriate errors."""
        with pytest.raises(ValueError):
            save_poses.to_dlc_file(
                request.getfixturevalue(invalid_pose_dataset),
                tmp_path / "test.h5",
            )
