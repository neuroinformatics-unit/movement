"""Test suite for the sample_data module."""

import pytest
from xarray import Dataset

from movement.sample_data import fetch_sample_data, list_sample_data


def test_list_sample_data():
    test_files = [
        "SLEAP_single-mouse_EPM.analysis.h5",
        "DLC_single-wasp.predictions.h5",
        "LP_mouse-face_AIND.predictions.csv",
    ]

    assert isinstance(list_sample_data(), list)
    assert all(file in list_sample_data() for file in test_files)


def test_fetch_sample_data():
    # SLEAP
    ds_sleap = fetch_sample_data("SLEAP_single-mouse_EPM.analysis.h5")
    assert isinstance(ds_sleap, Dataset) and ds_sleap.fps == 30

    # DeepLabCut
    ds_dlc = fetch_sample_data("DLC_single-wasp.predictions.h5")
    assert isinstance(ds_dlc, Dataset) and ds_dlc.fps == 40

    # LightningPose
    ds_lp = fetch_sample_data("LP_mouse-face_AIND.predictions.csv")
    assert isinstance(ds_lp, Dataset) and ds_lp.fps == 60

    # Test with an invalid file
    with pytest.raises(ValueError):
        fetch_sample_data("nonexistent_file")
