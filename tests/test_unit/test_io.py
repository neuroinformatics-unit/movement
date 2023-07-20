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
