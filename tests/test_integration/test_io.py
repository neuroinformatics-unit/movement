import h5py
import numpy as np
import pytest
import xarray as xr
from pytest import DATA_PATHS

from movement.io import load_poses, save_poses


@pytest.fixture(params=["dlc.h5", "dlc.csv"])
def dlc_output_file(request, tmp_path):
    """Return the output file path for a DLC .h5 or .csv file."""
    return tmp_path / request.param


def test_load_and_save_to_dlc_style_df(valid_dlc_poses_df):
    """Test that loading pose tracks from a DLC-style DataFrame and
    converting back to a DataFrame returns the same data values.
    """
    ds = load_poses.from_dlc_style_df(valid_dlc_poses_df)
    df = save_poses.to_dlc_style_df(ds, split_individuals=False)
    np.testing.assert_allclose(df.values, valid_dlc_poses_df.values)


def test_save_and_load_dlc_file(dlc_output_file, valid_poses_dataset):
    """Test that saving pose tracks to DLC .h5 and .csv files and then
    loading them back in returns the same Dataset.
    """
    save_poses.to_dlc_file(
        valid_poses_dataset, dlc_output_file, split_individuals=False
    )
    ds = load_poses.from_dlc_file(dlc_output_file)
    xr.testing.assert_allclose(ds, valid_poses_dataset)


def test_convert_sleap_to_dlc_file(sleap_file, dlc_output_file):
    """Test that pose tracks loaded from SLEAP .slp and .h5 files,
    when converted to DLC .h5 and .csv files and re-loaded return
    the same Datasets.
    """
    sleap_ds = load_poses.from_sleap_file(sleap_file)
    save_poses.to_dlc_file(sleap_ds, dlc_output_file, split_individuals=False)
    dlc_ds = load_poses.from_dlc_file(dlc_output_file)
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
    sleap_h5_file, fps, new_h5_file
):
    """Test that saving pose tracks (loaded from a SLEAP analysis
    file) to a SLEAP-style .h5 analysis file returns the same file
    contents.
    """
    sleap_h5_file_path = DATA_PATHS.get(sleap_h5_file)
    ds = load_poses.from_sleap_file(sleap_h5_file_path, fps=fps)
    save_poses.to_sleap_analysis_file(ds, new_h5_file)

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
def test_to_sleap_analysis_file_source_file(file, new_h5_file):
    """Test that saving pose tracks (loaded from valid source files)
    to a SLEAP-style .h5 analysis file stores the .slp labels path
    only when the source file is a .slp file.
    """
    file_path = DATA_PATHS.get(file)
    if file.startswith("DLC"):
        ds = load_poses.from_dlc_file(file_path)
    else:
        ds = load_poses.from_sleap_file(file_path)
    save_poses.to_sleap_analysis_file(ds, new_h5_file)

    with h5py.File(new_h5_file, "r") as f:
        if file_path.suffix == ".slp":
            assert file_path.name in f["labels_path"][()].decode()
        else:
            assert f["labels_path"][()].decode() == ""


def test_save_and_load_to_nwb_file(valid_poses_dataset):
    """Test that saving pose tracks to NWBFile and then loading
    the file back in returns the same Dataset.
    """
    nwb_files = save_poses.to_nwb_file(valid_poses_dataset)
    ds_singles = [load_poses.from_nwb_file(nwb_file) for nwb_file in nwb_files]
    ds = xr.merge(ds_singles)
    # Change expected differences to match valid_poses_dataset
    ds["time"] = ds.time.astype(int)
    ds.attrs["time_unit"] = valid_poses_dataset.attrs["time_unit"]
    ds.attrs["source_file"] = valid_poses_dataset.attrs["source_file"]
    del ds.attrs["fps"]
    xr.testing.assert_allclose(ds, valid_poses_dataset)
