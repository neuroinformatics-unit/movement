import numpy as np
import pytest
import xarray as xr

from movement.io import load_poses, save_poses


class TestPosesIO:
    """Test the IO functionalities of the PoseTracks class."""

    def test_load_and_save_to_dlc_df(self, dlc_style_df):
        """Test that loading pose tracks from a DLC-style DataFrame and
        converting back to a DataFrame returns the same data values."""
        ds = load_poses.from_dlc_df(dlc_style_df)
        df = save_poses.to_dlc_df(ds, split_individuals=False)
        np.testing.assert_allclose(df.values, dlc_style_df.values)

    @pytest.mark.parametrize("file_name", ["dlc.h5", "dlc.csv"])
    def test_save_and_load_dlc_file(
        self, file_name, valid_pose_dataset, tmp_path
    ):
        """Test that saving pose tracks to DLC .h5 and .csv files and then
        loading them back in returns the same Dataset."""
        save_poses.to_dlc_file(
            valid_pose_dataset, tmp_path / file_name, split_individuals=False
        )
        ds = load_poses.from_dlc_file(tmp_path / file_name)
        xr.testing.assert_allclose(ds, valid_pose_dataset)
