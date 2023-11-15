from contextlib import nullcontext as does_not_raise
from pathlib import Path

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
        """Return the file path for a new DeepLabCut H5 file."""
        return tmp_path / "new_dlc_file.h5"

    @pytest.fixture
    def new_dlc_csv_file(self, tmp_path):
        """Return the file path for a new DeepLabCut csv file."""
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
        ],
    )
    def test_to_dlc_df(self, ds, expected_exception):
        """Test that converting a valid/invalid xarray dataset to
        a DeepLabCut-style pandas DataFrame returns the expected result."""
        with expected_exception as e:
            df = save_poses.to_dlc_df(ds, split_individuals=False)
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
                split_individuals=False,
            )

    @pytest.mark.parametrize(
        "valid_pose_dataset, split_value",
        [("single_track_array", True), ("multi_track_array", False)],
        indirect=["valid_pose_dataset"],
    )
    def test_auto_split_individuals(self, valid_pose_dataset, split_value):
        """Test that setting 'split_individuals' to 'auto' yields True
        for single-individual datasets and False for multi-individual ones."""
        assert (
            save_poses._auto_split_individuals(valid_pose_dataset)
            == split_value
        )

    @pytest.mark.parametrize(
        "valid_pose_dataset, split_individuals",
        [
            ("single_track_array", True),  # single-individual, split
            ("multi_track_array", False),  # multi-individual, no split
            ("single_track_array", False),  # single-individual, no split
            ("multi_track_array", True),  # multi-individual, split
        ],
        indirect=["valid_pose_dataset"],
    )
    def test_to_dlc_df_split_individuals(
        self,
        valid_pose_dataset,
        split_individuals,
        request,
    ):
        """Test that the 'split_individuals' argument affects the behaviour
        of the 'to_dlc_df` function as expected
        """
        df = save_poses.to_dlc_df(valid_pose_dataset, split_individuals)
        # Get the names of the individuals in the dataset
        ds = request.getfixturevalue("valid_pose_dataset")
        ind_names = ds.individuals.values

        if split_individuals is False:
            # this should produce a single df in multi-animal DLC format
            assert isinstance(df, pd.DataFrame)
            assert df.columns.names == [
                "scorer",
                "individuals",
                "bodyparts",
                "coords",
            ]
            assert all(
                [ind in df.columns.get_level_values("individuals")]
                for ind in ind_names
            )
        elif split_individuals is True:
            # this should produce a dict of dfs in single-animal DLC format
            assert isinstance(df, dict)
            for ind in ind_names:
                assert ind in df.keys()
                assert isinstance(df[ind], pd.DataFrame)
                assert df[ind].columns.names == [
                    "scorer",
                    "bodyparts",
                    "coords",
                ]

    @pytest.mark.parametrize(
        "split_individuals, expected_exception",
        [
            (True, does_not_raise()),
            (False, does_not_raise()),
            ("auto", does_not_raise()),
            ("1", pytest.raises(ValueError, match="boolean or 'auto'")),
        ],
    )
    def test_to_dlc_file_split_individuals(
        self,
        valid_pose_dataset,
        new_dlc_h5_file,
        split_individuals,
        expected_exception,
        request,
    ):
        """Test that the 'split_individuals' argument affects the behaviour
        of the 'to_dlc_file` function as expected
        """

        with expected_exception:
            save_poses.to_dlc_file(
                valid_pose_dataset,
                new_dlc_h5_file,
                split_individuals,
            )
            ds = request.getfixturevalue("valid_pose_dataset")

            # "auto" becomes False, default valid dataset is multi-individual
            if split_individuals in [False, "auto"]:
                # this should save only one file
                assert new_dlc_h5_file.is_file()
                new_dlc_h5_file.unlink()
            elif split_individuals is True:
                # this should save one file per individual
                for ind in ds.individuals.values:
                    file_path_ind = Path(
                        f"{new_dlc_h5_file.with_suffix('')}_{ind}.h5"
                    )
                    assert file_path_ind.is_file()
                    file_path_ind.unlink()
