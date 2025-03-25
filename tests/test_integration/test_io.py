import h5py
import numpy as np
import pytest
import xarray as xr
from pytest import DATA_PATHS

from movement.io import load_dataset, save_dataset
from movement.io.save_dataset import to_dlc_style_df, to_dlc_file, to_sleap_analysis_file


class TestPosesIO:
    """Test the IO functionalities for poses."""

    @pytest.fixture(params=["dlc.h5", "dlc.csv"])
    def dlc_output_file(self, request, tmp_path):
        """Return the output file path for a DLC .h5 or .csv file."""
        return tmp_path / request.param

    def test_load_and_save_to_dlc_style_df(self, valid_dlc_poses_df):
        """Test that loading pose tracks from a DLC-style DataFrame and
        converting back to a DataFrame returns the same data values.
        """
        ds = load_dataset.from_dlc_style_df(valid_dlc_poses_df)
        df = to_dlc_style_df(ds, split_individuals=False)
        np.testing.assert_allclose(df.values, valid_dlc_poses_df.values)

    def test_save_and_load_dlc_file(
        self, dlc_output_file, valid_poses_dataset
    ):
        """Test that saving pose tracks to DLC .h5 and .csv files and then
        loading them back in returns the same Dataset.
        """
        save_dataset.to_dlc_file(
            valid_poses_dataset, dlc_output_file, split_individuals=False
        )
        ds = load_dataset.from_dlc_file(dlc_output_file)
        xr.testing.assert_allclose(ds, valid_poses_dataset)

    def test_convert_sleap_to_dlc_file(self, sleap_file, dlc_output_file):
        """Test that pose tracks loaded from SLEAP .slp and .h5 files,
        when converted to DLC .h5 and .csv files and re-loaded return
        the same Datasets.
        """
        sleap_ds = load_dataset.from_sleap_file(sleap_file)
        save_dataset.to_dlc_file(
            sleap_ds, dlc_output_file, split_individuals=False
        )
        dlc_ds = load_dataset.from_dlc_file(dlc_output_file)
        xr.testing.assert_allclose(sleap_ds, dlc_ds)

    @pytest.mark.parametrize(
        "sleap_h5_file, fps",
        [
            ("SLEAP_single-mouse_EPM.analysis.h5", 30),
            ("SLEAP_three-mice_Aeon_proofread.analysis.h5", None),
            ("SLEAP_three-mice_Aeon_mixed-labels.analysis.h5", 50),
        ],
    )
    def test_to_sleap_analysis_file_returns_same_h5_file_content(
        self, sleap_h5_file, fps, new_h5_file
    ):
        """Test that saving pose tracks (loaded from a SLEAP analysis
        file) to a SLEAP-style .h5 analysis file returns the same file
        contents.
        """
        sleap_h5_file_path = DATA_PATHS.get(sleap_h5_file)
        ds = load_dataset.from_sleap_file(sleap_h5_file_path, fps=fps)
        save_dataset.to_sleap_analysis_file(ds, new_h5_file)

        with (
            h5py.File(ds.source_file, "r") as file_in,
            h5py.File(new_h5_file, "r") as file_out,
        ):
            assert set(file_in.keys()) == set(file_out.keys())
            keys = [
                "track_occupancy",
                "tracks",
                "point_scores",
            ]
            for key in keys:
                np.testing.assert_allclose(file_in[key][:], file_out[key][:])

    @pytest.mark.parametrize(
        "file",
        [
            "DLC_single-wasp.predictions.h5",
            "DLC_two-mice.predictions.csv",
            "SLEAP_single-mouse_EPM.analysis.h5",
            "SLEAP_three-mice_Aeon_proofread.predictions.slp",
        ],
    )
    def test_to_sleap_analysis_file_source_file(self, file, new_h5_file):
        """Test that saving pose tracks (loaded from valid source files)
        to a SLEAP-style .h5 analysis file stores the .slp labels path
        only when the source file is a .slp file.
        """
        file_path = DATA_PATHS.get(file)
        if file.startswith("DLC"):
            ds = load_dataset.from_dlc_file(file_path)
        else:
            ds = load_dataset.from_sleap_file(file_path)
        save_dataset.to_sleap_analysis_file(ds, new_h5_file)

        with h5py.File(new_h5_file, "r") as f:
            if file_path.suffix == ".slp":
                assert file_path.name in f["labels_path"][()].decode()
            else:
                assert f["labels_path"][()].decode() == ""

def test_dlc_style_df_roundtrip(valid_dlc_poses_df):
    """Test roundtrip conversion between DeepLabCut-style DataFrame and Dataset."""
    ds = load_dataset.from_dlc_style_df(valid_dlc_poses_df)
    df = to_dlc_style_df(ds, split_individuals=False)
    pd.testing.assert_frame_equal(df, valid_dlc_poses_df)

def test_dlc_file_roundtrip(valid_dlc_poses_df, tmp_path):
    """Test roundtrip conversion between DeepLabCut file and Dataset."""
    dlc_output_file = tmp_path / "test.h5"
    save_dataset.to_dlc_file(
        load_dataset.from_dlc_style_df(valid_dlc_poses_df),
        dlc_output_file,
    )
    ds = load_dataset.from_dlc_file(dlc_output_file)
    assert isinstance(ds, xr.Dataset)
    assert "time" in ds.dims
    assert "keypoint" in ds.dims
    assert "individual" in ds.dims

def test_sleap_to_dlc_roundtrip(sleap_file, tmp_path):
    """Test roundtrip conversion between SLEAP file and DeepLabCut file."""
    sleap_ds = load_dataset.from_sleap_file(sleap_file)
    dlc_output_file = tmp_path / "test.h5"
    save_dataset.to_dlc_file(
        sleap_ds,
        dlc_output_file,
    )
    dlc_ds = load_dataset.from_dlc_file(dlc_output_file)
    assert isinstance(dlc_ds, xr.Dataset)
    assert "time" in dlc_ds.dims
    assert "keypoint" in dlc_ds.dims
    assert "individual" in dlc_ds.dims

def test_sleap_analysis_file_roundtrip(sleap_h5_file_path, tmp_path):
    """Test roundtrip conversion between SLEAP analysis file and Dataset."""
    fps = 30
    ds = load_dataset.from_sleap_file(sleap_h5_file_path, fps=fps)
    new_h5_file = tmp_path / "test.h5"
    save_dataset.to_sleap_analysis_file(ds, new_h5_file)
    assert new_h5_file.exists()

def test_file_roundtrip(file_path):
    """Test roundtrip conversion between different file formats."""
    ds = load_dataset.from_dlc_file(file_path)
    new_h5_file = file_path.parent / "test.h5"
    save_dataset.to_sleap_analysis_file(ds, new_h5_file)
    assert new_h5_file.exists()
