import os

import h5py
import numpy as np
import pandas as pd
import pytest
from xarray.testing import assert_allclose

from movement.datasets import fetch_pose_data_path
from movement.io import PoseTracks


class TestPoseTracksIO:
    """Test the IO functionalities of the PoseTracks class."""

    @pytest.fixture
    def dlc_file_h5_single(self):
        """Return the path to a valid DLC h5 file containing pose data
        for a single animal."""
        return fetch_pose_data_path("DLC_single-wasp.predictions.h5")

    @pytest.fixture
    def dlc_file_csv_single(self):
        """Return the path to a valid DLC .csv file containing pose data
        for a single animal. The underlying data is the same as in the
        `dlc_file_h5_single` fixture."""
        return fetch_pose_data_path("DLC_single-wasp.predictions.csv")

    @pytest.fixture
    def dlc_file_csv_multi(self):
        """Return the path to a valid DLC .csv file containing pose data
        for multiple animals."""
        return fetch_pose_data_path("DLC_two-mice.predictions.csv")

    @pytest.fixture
    def sleap_file_h5_single(self):
        """Return the path to a valid SLEAP "analysis" .h5 file containing
        pose data for a single animal."""
        return fetch_pose_data_path("SLEAP_single-mouse_EPM.analysis.h5")

    @pytest.fixture
    def sleap_file_slp_single(self):
        """Return the path to a valid SLEAP .slp file containing
        predicted poses (labels) for a single animal."""
        return fetch_pose_data_path("SLEAP_single-mouse_EPM.predictions.slp")

    @pytest.fixture
    def sleap_file_h5_multi(self):
        """Return the path to a valid SLEAP "analysis" .h5 file containing
        pose data for multiple animals."""
        return fetch_pose_data_path(
            "SLEAP_three-mice_Aeon_proofread.analysis.h5"
        )

    @pytest.fixture
    def sleap_file_slp_multi(self):
        """Return the path to a valid SLEAP .slp file containing
        predicted poses (labels) for multiple animals."""
        return fetch_pose_data_path(
            "SLEAP_three-mice_Aeon_proofread.predictions.slp"
        )

    @pytest.fixture
    def valid_dlc_files(
        dlc_file_h5_single, dlc_file_csv_single, dlc_file_csv_multi
    ):
        """Aggregate all valid DLC files in a dictionary, for convenience."""
        return {
            "h5_single": dlc_file_h5_single,
            "csv_single": dlc_file_csv_single,
            "csv_multi": dlc_file_csv_multi,
        }

    @pytest.fixture
    def valid_sleap_files(
        sleap_file_h5_single,
        sleap_file_slp_single,
        sleap_file_h5_multi,
        sleap_file_slp_multi,
    ):
        """Aggregate all valid SLEAP files in a dictionary, for convenience."""
        return {
            "h5_single": sleap_file_h5_single,
            "slp_single": sleap_file_slp_single,
            "h5_multi": sleap_file_h5_multi,
            "slp_multi": sleap_file_slp_multi,
        }

    @pytest.fixture
    def invalid_files(self, tmp_path):
        unreadable_file = tmp_path / "unreadable.h5"
        with open(unreadable_file, "w") as f:
            f.write("unreadable data")
            os.chmod(f.name, 0o000)

        wrong_ext_file = tmp_path / "wrong_extension.txt"
        with open(wrong_ext_file, "w") as f:
            f.write("")

        h5_file_no_dataframe = tmp_path / "no_dataframe.h5"
        with h5py.File(h5_file_no_dataframe, "w") as f:
            f.create_dataset("data_in_list", data=[1, 2, 3])

        nonexistent_file = tmp_path / "nonexistent.h5"

        return {
            "unreadable": unreadable_file,
            "wrong_ext": wrong_ext_file,
            "no_dataframe": h5_file_no_dataframe,
            "nonexistent": nonexistent_file,
        }

    @pytest.fixture
    def dlc_style_df(self, dlc_file_h5_single):
        """Return a valid DLC-style DataFrame."""
        df = pd.read_hdf(dlc_file_h5_single)
        return df

    def test_load_from_dlc_file_csv_or_h5_file_returns_same(
        self, dlc_file_h5_single, dlc_file_csv_single
    ):
        """Test that loading pose tracks from DLC .csv and .h5 files
        return the same Dataset."""
        ds_from_h5 = PoseTracks.from_dlc_file(dlc_file_h5_single)
        ds_from_csv = PoseTracks.from_dlc_file(dlc_file_csv_single)
        assert_allclose(ds_from_h5, ds_from_csv)

    @pytest.mark.parametrize("fps", [None, -5, 0, 30, 60.0])
    def test_fps_and_time_coords(self, sleap_file_h5_multi, fps):
        """Test that time coordinates are set according to the fps."""
        ds = PoseTracks.from_sleap_file(sleap_file_h5_multi, fps=fps)
        if (fps is None) or (fps <= 0):
            assert ds.fps is None
            assert ds.time_unit == "frames"
        else:
            assert ds.fps == fps
            assert ds.time_unit == "seconds"
            np.allclose(
                ds.coords["time"].data,
                np.arange(ds.dims["time"], dtype=int) / ds.attrs["fps"],
            )

    def test_from_and_to_dlc_df(self, dlc_style_df):
        """Test that loading pose tracks from a DLC-style DataFrame and
        converting back to a DataFrame returns the same data values."""
        ds = PoseTracks.from_dlc_df(dlc_style_df)
        df = ds.to_dlc_df()
        assert np.allclose(df.values, dlc_style_df.values)

    def test_load_from_str_path(self, sleap_file_h5_single):
        """Test that file paths provided as strings are accepted as input."""
        assert_allclose(
            PoseTracks.from_sleap_file(sleap_file_h5_single),
            PoseTracks.from_sleap_file(sleap_file_h5_single.as_posix()),
        )

    @pytest.mark.parametrize(
        "scores_array", [None, np.zeros((10, 2, 2)), np.zeros((10, 2, 3))]
    )
    def test_init_scores(self, scores_array):
        """Test that confidence scores are correctly initialized."""
        tracks = np.random.rand(10, 2, 2, 2)

        if scores_array is None:
            ds = PoseTracks(tracks, scores_array=scores_array)
            assert ds.confidence_scores.shape == (10, 2, 2)
            assert np.all(np.isnan(ds.confidence_scores.data))
        elif scores_array.shape == (10, 2, 2):
            ds = PoseTracks(tracks, scores_array=scores_array)
            assert np.allclose(ds.confidence_scores.data, scores_array)
        else:
            with pytest.raises(ValueError):
                ds = PoseTracks(tracks, scores_array=scores_array)

    @pytest.mark.parametrize(
        "individual_names",
        [None, ["animal_1", "animal_2"], ["animal_1", "animal_2", "animal_3"]],
    )
    def test_init_individual_names(self, individual_names):
        """Test that individual names are correctly initialized."""
        tracks = np.random.rand(10, 2, 2, 2)

        if individual_names is None:
            ds = PoseTracks(tracks, individual_names=individual_names)
            assert ds.dims["individuals"] == 2
            assert all(
                [
                    f"individual_{i}" in ds.coords["individuals"]
                    for i in range(2)
                ]
            )
        elif len(individual_names) == 2:
            ds = PoseTracks(tracks, individual_names=individual_names)
            assert ds.dims["individuals"] == 2
            assert all(
                [n in ds.coords["individuals"] for n in individual_names]
            )
        else:
            with pytest.raises(ValueError):
                ds = PoseTracks(tracks, individual_names=individual_names)

    @pytest.mark.parametrize(
        "keypoint_names", [None, ["kp_1", "kp_2"], ["kp_1", "kp_2", "kp_3"]]
    )
    def test_init_keypoint_names(self, keypoint_names):
        """Test that keypoint names are correctly initialized."""
        tracks = np.random.rand(10, 2, 2, 2)

        if keypoint_names is None:
            ds = PoseTracks(tracks, keypoint_names=keypoint_names)
            assert ds.dims["keypoints"] == 2
            assert all(
                [f"keypoint_{i}" in ds.coords["keypoints"] for i in range(2)]
            )
        elif len(keypoint_names) == 2:
            ds = PoseTracks(tracks, keypoint_names=keypoint_names)
            assert ds.dims["keypoints"] == 2
            assert all([n in ds.coords["keypoints"] for n in keypoint_names])
        else:
            with pytest.raises(ValueError):
                ds = PoseTracks(tracks, keypoint_names=keypoint_names)
