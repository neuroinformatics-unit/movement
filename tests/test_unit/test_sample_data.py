"""Test suite for the sample_data module."""

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pooch
import pytest
from requests.exceptions import RequestException
from xarray import Dataset

from movement.sample_data import (
    SAMPLE_DATA,
    _fetch_and_unzip,
    _fetch_metadata,
    fetch_dataset,
    fetch_dataset_paths,
    hide_pooch_hash_logs,
    list_datasets,
)

# Define sample datasets for parametrization
SAMPLE_DATASETS = {
    "SLEAP_three-mice_Aeon_proofread.analysis.h5": {
        "fps": 50,
        "frame_file": "three-mice_Aeon_frame-5sec.png",
        "video_file": "three-mice_Aeon_video.avi",
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
    "VIA_multiple-crabs_5-frames_labels.csv": {
        "fps": None,
        "frame_file": None,
        "video_file": None,
    },
    "TRex_five-locusts.zip": {
        "fps": 5,
        "frame_file": "locusts-noqr_n-5_date-20250117_frame-bg.png",
        "video_file": None,
    },
}


@pytest.fixture(scope="module")
def sample_dataset_names():
    """Return a list of sample dataset names."""
    return list(SAMPLE_DATASETS.keys())


@pytest.fixture(params=list(SAMPLE_DATASETS.items()))
def sample_dataset(request):
    """Return the name of a sample dataset and its metadata."""
    sample_name, sample_metadata = request.param
    return sample_name, sample_metadata


def validate_metadata(metadata: dict[str, dict]) -> None:
    """Assert that the metadata is in the expected format."""
    metadata_fields = [
        "sha256sum",
        "type",
        "source_software",
        "type",
        "fps",
        "species",
        "number_of_individuals",
        "shared_by",
        "frame",
        "video",
        "note",
    ]
    check_yaml_msg = "Check the format of the metadata .yaml file."
    assert isinstance(metadata, dict), (
        f"Expected metadata to be a dictionary. {check_yaml_msg}"
    )
    assert all(isinstance(ds, str) for ds in metadata), (
        f"Expected metadata keys to be strings. {check_yaml_msg}"
    )
    assert all(isinstance(val, dict) for val in metadata.values()), (
        f"Expected metadata values to be dicts. {check_yaml_msg}"
    )
    assert all(
        set(val.keys()) == set(metadata_fields) for val in metadata.values()
    ), f"Found issues with the names of metadata fields. {check_yaml_msg}"

    # check that metadata keys (file names) are unique
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


def test_list_datasets(sample_dataset_names):
    assert isinstance(list_datasets(), list)
    assert all(file in list_datasets() for file in sample_dataset_names)


@pytest.mark.parametrize("with_video", [True, False])
def test_fetch_dataset(sample_dataset, with_video):
    """Test fetch_dataset for each sample dataset with/without video."""
    sample_name, sample_metadata = sample_dataset
    # Skip TRex datasets as they are not supported yet
    if sample_name.startswith("TRex"):
        pytest.xfail("TRex datasets are not supported yet.")

    ds = fetch_dataset(sample_name, with_video=with_video)
    assert isinstance(ds, Dataset)

    # Check fps attribute
    assert getattr(ds, "fps", None) == sample_metadata["fps"]

    # Check frame_path
    frame_path = getattr(ds, "frame_path", None)
    if sample_metadata["frame_file"]:
        assert frame_path is not None
        assert frame_path.split("/")[-1] == sample_metadata["frame_file"]
    else:
        assert frame_path is None

    # Check video_path
    video_path = getattr(ds, "video_path", None)
    if sample_metadata["video_file"] and with_video:
        assert video_path is not None
        assert video_path.split("/")[-1] == sample_metadata["video_file"]
    else:
        assert video_path is None


@pytest.mark.parametrize(
    "sample_name, expected_exception",
    [
        ("nonexistent-dataset", ValueError),
        ("TRex_five-locusts.zip", NotImplementedError),
    ],
    ids=["invalid_file", "TRex_folder_zip"],
)
def test_fetch_dataset_invalid(sample_name, expected_exception):
    with pytest.raises(expected_exception):
        fetch_dataset(sample_name)


def test_fetch_dataset_paths(sample_dataset):
    """Test that the returned pose paths points to correct location.

    If the pose files are in a zipped folder, the path should point to the
    unzipped folder, otherwise it should point to the file itself.
    """
    sample_name, _ = sample_dataset
    paths = fetch_dataset_paths(sample_name)
    data_path = Path(paths.get("poses", paths.get("bboxes")))

    if sample_name.endswith(".zip"):
        # If the sample is a zip file,
        # the path should point to the unzipped folder
        assert data_path.is_dir()
        assert data_path.name == sample_name.replace(".zip", "")
        file_paths = list(data_path.iterdir())
        assert len(file_paths) > 1
        # Make sure unwanted files are not present
        assert not any(
            file_path.name in {".DS_Store", "Thumbs.db", "desktop.ini"}
            for file_path in file_paths
        )
    else:
        # If the sample is a single file,
        # the path should point to the file itself
        assert data_path.is_file()
        assert data_path.name == sample_name


@pytest.mark.parametrize(
    "failing_func,expected_warning",
    [
        pytest.param(
            "copy2",
            "Failed to copy file",
            id="copy2_exception",
        ),
        pytest.param(
            "rmtree",
            "Failed to remove unzip dir",
            id="rmtree_exception",
        ),
    ],
)
def test_fetch_and_unzip_exceptions(
    tmp_path, monkeypatch, caplog, failing_func, expected_warning
):
    """Test _fetch_and_unzip handles failures to copy files
    or remove dirs gracefully, raising the appropriate warning.

    It also tests that unwanted files are not copied to the new dest folder.
    """
    # Create the proper directory structure that the function expects
    extract_dir = tmp_path / "poses" / "test.zip.unzip" / "test"
    extract_dir.mkdir(parents=True, exist_ok=True)
    # Create two files: one valid and one unwanted
    file_names = ["file.txt", ".DS_Store"]
    for file_name in file_names:
        (extract_dir / file_name).touch()

    # Patch SAMPLE_DATA.path to use our tmp_path
    monkeypatch.setattr(SAMPLE_DATA, "path", tmp_path)
    # Patch SAMPLE_DATA.fetch to return the fake files
    monkeypatch.setattr(
        SAMPLE_DATA,
        "fetch",
        lambda *a, **k: [str(extract_dir / file) for file in file_names],
    )

    with patch(f"shutil.{failing_func}", side_effect=OSError("failed")):
        result = _fetch_and_unzip("poses", "test.zip")
        result_names = [p.name for p in result.iterdir()]
        if failing_func == "copy2":
            # original folder is kept, incl. unwanted file
            assert result == extract_dir
            assert ".DS_Store" in result_names
        else:  # rmtree
            # new dest folder is returned, without unwanted file
            assert result == tmp_path / "poses" / "test"
            assert ".DS_Store" not in result_names
        assert expected_warning in caplog.records[-1].getMessage()


@pytest.mark.parametrize(
    "message, should_block",
    [
        ("SHA256 hash of downloaded file: abc123", True),
        ("Use this value as the 'known_hash' argument", True),
        ("Downloading file from ", False),
    ],
)
def test_hide_pooch_hash_logs(message, should_block):
    """Test that ``HashFilter`` blocks only hash-related messages."""
    with hide_pooch_hash_logs():
        logger = pooch.get_logger()
        hash_filter = logger.filters[-1]

        # Create a mock log record
        record = logging.LogRecord(
            name="pooch",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg=message,
            args=(),
            exc_info=None,
        )

        # The filter returns False to block, True to allow
        message_allowed = hash_filter.filter(record)

        if should_block:
            assert not message_allowed, (
                f"Expected message to be blocked but was allowed: '{message}'"
            )
        else:
            assert message_allowed, (
                f"Expected message to be allowed but was blocked: '{message}'"
            )


@pytest.mark.parametrize("raise_exception", [True, False])
def test_hash_filter_removed_after_context(raise_exception):
    """Test that the hash filter applied by ``hide_pooch_hash_logs``
    is properly removed after exiting context, even when an exception occurs.
    """
    logger = pooch.get_logger()
    initial_filter_count = len(logger.filters)

    if raise_exception:
        with pytest.raises(ValueError), hide_pooch_hash_logs():
            assert len(logger.filters) == initial_filter_count + 1
            raise ValueError("Test exception")
    else:
        with hide_pooch_hash_logs():
            assert len(logger.filters) == initial_filter_count + 1

    # Filter should be removed after context exits
    assert len(logger.filters) == initial_filter_count
