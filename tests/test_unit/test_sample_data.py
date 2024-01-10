"""Test suite for the sample_data module."""

from unittest.mock import MagicMock, patch

import pooch
import pytest
from requests.exceptions import RequestException
from xarray import Dataset

from movement.sample_data import (
    _fetch_metadata,
    fetch_sample_data,
    list_sample_data,
)


@pytest.fixture(scope="module")
def valid_file_names_with_fps():
    """Return a dict containing one valid file name and the corresponding fps
    for each supported pose estimation tool."""
    return {
        "SLEAP_single-mouse_EPM.analysis.h5": 30,
        "DLC_single-wasp.predictions.h5": 40,
        "LP_mouse-face_AIND.predictions.csv": 60,
    }


def validate_metadata(metadata: list[dict]) -> None:
    """Assert that the metadata is in the expected format."""
    metadata_fields = [
        "file_name",
        "sha256sum",
        "source_software",
        "fps",
        "species",
        "number_of_individuals",
        "shared_by",
        "video_frame_file",
        "note",
    ]
    check_yaml_msg = "Check the format of the metadata yaml file."
    assert isinstance(
        metadata, list
    ), f"Expected metadata to be a list. {check_yaml_msg}"
    assert all(
        isinstance(file, dict) for file in metadata
    ), f"Expected metadata entries to be dicts. {check_yaml_msg}"
    assert all(
        set(file.keys()) == set(metadata_fields) for file in metadata
    ), f"Expected all metadata entries to have the same keys. {check_yaml_msg}"

    # check that filenames are unique
    file_names = [file["file_name"] for file in metadata]
    assert len(file_names) == len(set(file_names))

    # check that the first 3 fields are present and are strings
    required_fields = metadata_fields[:3]
    assert all(
        (isinstance(file[field], str))
        for file in metadata
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
    an error is raised."""
    metadata_file_name = "poses_files_metadata.yaml"
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
