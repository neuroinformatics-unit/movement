"""Test suite for the load_poses module."""

import pytest
from xarray import Dataset

from movement.sample_data import fetch_sample_data


def test_fetch_sample_data():
    # SLEAP
    ds_sleap = fetch_sample_data("SLEAP_single-mouse_EPM.analysis.h5")
    assert isinstance(ds_sleap, Dataset)

    # DeepLabCut
    ds_dlc = fetch_sample_data("DLC_single-wasp.predictions.h5")
    assert isinstance(ds_dlc, Dataset)

    # LightningPose
    ds_lp = fetch_sample_data("LP_mouse-face_AIND.predictions.csv")
    assert isinstance(ds_lp, Dataset)

    # Test with an invalid file
    with pytest.raises(ValueError):
        fetch_sample_data("nonexistent_file")
