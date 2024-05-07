"""Test suite for the sample_data module."""

from unittest.mock import MagicMock, patch

import pooch
import pytest
from requests.exceptions import RequestException
from xarray import Dataset

from movement.sample_data import _fetch_metadata, fetch_dataset, list_datasets


@pytest.fixture(scope="module")
def valid_sample_datasets():
    """Return a dict mapping valid sample dataset file names to their
    respective fps values, and associated frame and video file names.
    """
    return {
        "SLEAP_single-mouse_EPM.analysis.h5": {
            "fps": 30,
            "frame_file": "single-mouse_EPM_frame-20sec.png",
            "video_file": "single-mouse_EPM_video.mp4",
        },
        "DLC_single-wasp.predictions.h5": {
            "fps": 40,
            "frame_file": "single-wasp_frame-10sec.png",
            "video_file": None,
        },
        "LP_mouse-face_AIND.predictions.csv": {
            "fps": 60,
            "frame_file": None,
            "video_file": None,
        },
    }


def validate_metadata(metadata: dict[str, dict]) -> None:
    """Assert that the metadata is in the expected format."""
    metadata_fields = [
        "sha256sum",
        "source_software",
        "fps",
        "species",
        "number_of_individuals",
        "shared_by",
        "frame",
        "video",
        "note",
    ]
    check_yaml_msg = "Check the format of the metadata .yaml file."
    assert isinstance(
        metadata, dict
    ), f"Expected metadata to be a dictionary. {check_yaml_msg}"
    assert all(
        isinstance(ds, str) for ds in metadata
    ), f"Expected metadata keys to be strings. {check_yaml_msg}"
    assert all(
        isinstance(val, dict) for val in metadata.values()
    ), f"Expected metadata values to be dicts. {check_yaml_msg}"
    assert all(
        set(val.keys()) == set(metadata_fields) for val in metadata.values()
    ), f"Found issues with the names of medatada fields. {check_yaml_msg}"

    # check that metadata keys (pose file names) are unique
    assert len(metadata.keys()) == len(set(metadata.keys()))

    # check that the first 2 fields are present and are strings
    required_fields = metadata_fields[:2]
    assert all(
        (isinstance(val[field], str))
        for val in metadata.values()
        for field in required_fields
    )


# Mock pooch.retrieve with RequestException as side_effect
mock_retrieve = MagicMock(pooch.retrieve, side_effect=RequestException)


@pytest.mark.parametrize("download_fails", [True, False])
@pytest.mark.parametrize("local_exists", [True, False])
def test_fetch_metadata(tmp_path, caplog, download_fails, local_exists):
    """Test the fetch_metadata function with different combinations of
    failed download and pre-existing local file. The expected behavior is
    that the function will try to download the metadata file, and if that
    fails, it will try to load an existing local file. If neither succeeds,
    an error is raised.
    """
    metadata_file_name = "metadata.yaml"
    local_file_path = tmp_path / metadata_file_name

    with patch("movement.sample_data.DATA_DIR", tmp_path):
        # simulate the existence of a local metadata file
        if local_exists:
            local_file_path.touch()

        if download_fails:
            # simulate a failed download
            with patch("movement.sample_data.pooch.retrieve", mock_retrieve):
                if local_exists:
                    _fetch_metadata(metadata_file_name)
                    # check that a warning was logged
                    assert (
                        "Will use the existing local version instead"
                        in caplog.records[-1].getMessage()
                    )
                else:
                    with pytest.raises(
                        RequestException, match="Failed to download"
                    ):
                        _fetch_metadata(metadata_file_name, data_dir=tmp_path)
        else:
            metadata = _fetch_metadata(metadata_file_name, data_dir=tmp_path)
            assert local_file_path.is_file()
            validate_metadata(metadata)


def test_list_datasets(valid_sample_datasets):
    assert isinstance(list_datasets(), list)
    assert all(file in list_datasets() for file in valid_sample_datasets)


def test_fetch_dataset(valid_sample_datasets):
    # test with valid files
    for sample_name, sample in valid_sample_datasets.items():
        ds = fetch_dataset(sample_name)
        assert isinstance(ds, Dataset)

        assert ds.attrs["fps"] == sample["fps"]

        if sample["frame_file"]:
            assert ds.attrs["frame_path"].name == sample["frame_file"]
        if sample["video_file"]:
            assert ds.attrs["video_path"].name == sample["video_file"]

    # Test with an invalid file
    with pytest.raises(ValueError):
        fetch_dataset("nonexistent_file")
