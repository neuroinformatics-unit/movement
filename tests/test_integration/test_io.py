import numpy as np
import pytest
import xarray as xr
from pytest import POSE_DATA

from movement.io import load_poses, save_poses


class TestPosesIO:
    """Test the IO functionalities of the PoseTracks class."""

    @pytest.fixture(
        params=[
            "SLEAP_single-mouse_EPM.analysis.h5",
            "SLEAP_single-mouse_EPM.predictions.slp",
            "SLEAP_three-mice_Aeon_proofread.analysis.h5",
            "SLEAP_three-mice_Aeon_proofread.predictions.slp",
        ]
    )
    def sleap_file(self, request):
        """Return the file path for a SLEAP .h5 or .slp file."""
        return POSE_DATA.get(request.param)

    @pytest.fixture(params=["dlc.h5", "dlc.csv"])
    def dlc_output_file(self, request, tmp_path):
        """Return the output file path for a DLC .h5 or .csv file."""
        return tmp_path / request.param

    def test_load_and_save_to_dlc_df(self, dlc_style_df):
        """Test that loading pose tracks from a DLC-style DataFrame and
        converting back to a DataFrame returns the same data values."""
        ds = load_poses.from_dlc_df(dlc_style_df)
        df = save_poses.to_dlc_df(ds)
        np.testing.assert_allclose(df.values, dlc_style_df.values)

    def test_save_and_load_dlc_file(self, dlc_output_file, valid_pose_dataset):
        """Test that saving pose tracks to DLC .h5 and .csv files and then
        loading them back in returns the same Dataset."""
        save_poses.to_dlc_file(valid_pose_dataset, dlc_output_file)
        ds = load_poses.from_dlc_file(dlc_output_file)
        xr.testing.assert_allclose(ds, valid_pose_dataset)

    def test_convert_sleap_to_dlc_file(self, sleap_file, dlc_output_file):
        """Test that pose tracks loaded from SLEAP .slp and .h5 files,
        when converted to DLC .h5 and .csv files and re-loaded return
        the same Datasets."""
        sleap_ds = load_poses.from_sleap_file(sleap_file)
        save_poses.to_dlc_file(sleap_ds, dlc_output_file)
        dlc_ds = load_poses.from_dlc_file(dlc_output_file)
        xr.testing.assert_allclose(sleap_ds, dlc_ds)
