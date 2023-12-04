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

    output_files = [
        {
            "file_fixture": "fake_h5_file",
            "to_dlc_file_expected_exception": pytest.raises(FileExistsError),
            "to_sleap_file_expected_exception": pytest.raises(FileExistsError),
            # invalid file path
        },
        {
            "file_fixture": "directory",
            "to_dlc_file_expected_exception": pytest.raises(IsADirectoryError),
            "to_sleap_file_expected_exception": pytest.raises(
                IsADirectoryError
            ),
            # invalid file path
        },
        {
            "file_fixture": "new_file_wrong_ext",
            "to_dlc_file_expected_exception": pytest.raises(ValueError),
            "to_sleap_file_expected_exception": pytest.raises(ValueError),
            # invalid file path
        },
        {
            "file_fixture": "new_csv_file",
            "to_dlc_file_expected_exception": does_not_raise(),
            "to_sleap_file_expected_exception": pytest.raises(ValueError),
            # valid file path for dlc, invalid for sleap
        },
        {
            "file_fixture": "new_h5_file",
            "to_dlc_file_expected_exception": does_not_raise(),
            "to_sleap_file_expected_exception": does_not_raise(),
            # valid file path
        },
    ]

    @pytest.fixture(params=output_files)
    def output_file_params(self, request):
        """Return a dictionary containing parameters for testing saving
        valid pose datasets to DeepLabCut- or SLEAP-style files."""
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

    def test_to_dlc_file_valid_dataset(
        self, output_file_params, valid_pose_dataset, request
    ):
        """Test that saving a valid pose dataset to a valid/invalid
        DeepLabCut-style file returns the appropriate errors."""
        with output_file_params.get("to_dlc_file_expected_exception"):
            file_fixture = output_file_params.get("file_fixture")
            val = request.getfixturevalue(file_fixture)
            file_path = val.get("file_path") if isinstance(val, dict) else val
            save_poses.to_dlc_file(valid_pose_dataset, file_path)

    def test_to_dlc_file_invalid_dataset(self, invalid_pose_dataset, tmp_path):
        """Test that saving an invalid pose dataset to a valid
        DeepLabCut-style file returns the appropriate errors."""
        with pytest.raises(ValueError):
            save_poses.to_dlc_file(
                invalid_pose_dataset,
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
    ):
        """Test that the `split_individuals` argument affects the behaviour
        of the `to_dlc_df` function as expected
        """
        df = save_poses.to_dlc_df(valid_pose_dataset, split_individuals)
        # Get the names of the individuals in the dataset
        ind_names = valid_pose_dataset.individuals.values
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
        new_h5_file,
        split_individuals,
        expected_exception,
    ):
        """Test that the `split_individuals` argument affects the behaviour
        of the `to_dlc_file` function as expected.
        """
        with expected_exception:
            save_poses.to_dlc_file(
                valid_pose_dataset,
                new_h5_file,
                split_individuals,
            )
            # Get the names of the individuals in the dataset
            ind_names = valid_pose_dataset.individuals.values
            # "auto" becomes False, default valid dataset is multi-individual
            if split_individuals in [False, "auto"]:
                # this should save only one file
                assert new_h5_file.is_file()
                new_h5_file.unlink()
            elif split_individuals is True:
                # this should save one file per individual
                for ind in ind_names:
                    file_path_ind = Path(
                        f"{new_h5_file.with_suffix('')}_{ind}.h5"
                    )
                    assert file_path_ind.is_file()
                    file_path_ind.unlink()

    def test_to_sleap_analysis_file_valid_dataset(
        self, output_file_params, valid_pose_dataset, request
    ):
        """Test that saving a valid pose dataset to a valid/invalid
        SLEAP-style file returns the appropriate errors."""
        with output_file_params.get("to_sleap_file_expected_exception"):
            file_fixture = output_file_params.get("file_fixture")
            val = request.getfixturevalue(file_fixture)
            file_path = val.get("file_path") if isinstance(val, dict) else val
            save_poses.to_sleap_analysis_file(valid_pose_dataset, file_path)

    def test_to_sleap_analysis_file_invalid_dataset(
        self, invalid_pose_dataset, new_h5_file
    ):
        """Test that saving an invalid pose dataset to a valid
        SLEAP-style file returns the appropriate errors."""
        with pytest.raises(ValueError):
            save_poses.to_sleap_analysis_file(
                invalid_pose_dataset,
                new_h5_file,
            )

    def test_remove_unoccupied_tracks(self, valid_pose_dataset):
        """Test that removing unoccupied tracks from a valid pose dataset
        returns the expected result."""
        new_individuals = [f"ind{i}" for i in range(1, 4)]
        # Add new individual with NaN data
        ds = valid_pose_dataset.reindex(individuals=new_individuals)
        ds = save_poses._remove_unoccupied_tracks(ds)
        xr.testing.assert_equal(ds, valid_pose_dataset)
