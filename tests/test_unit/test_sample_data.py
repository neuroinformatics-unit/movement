"""Test suite for the sample_data module."""

import pytest
from xarray import Dataset

from movement.sample_data import fetch_sample_data, list_sample_data


@pytest.fixture(scope="module")
def valid_file_names_with_fps():
    """Return a dict containing one valid file name and the corresponding fps
    for each supported pose estimation tool."""
    return {
        "SLEAP_single-mouse_EPM.analysis.h5": 30,
        "DLC_single-wasp.predictions.h5": 40,
        "LP_mouse-face_AIND.predictions.csv": 60,
    }


def test_list_sample_data(valid_file_names_with_fps):
    assert isinstance(list_sample_data(), list)
    assert all(
        file in list_sample_data() for file in valid_file_names_with_fps
    )


def test_fetch_sample_data(valid_file_names_with_fps):
    # test with valid files
    for file, fps in valid_file_names_with_fps.items():
        ds = fetch_sample_data(file)
        assert isinstance(ds, Dataset) and ds.fps == fps

    # Test with an invalid file
    with pytest.raises(ValueError):
        fetch_sample_data("nonexistent_file")
