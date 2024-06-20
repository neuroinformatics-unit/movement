"""Test suite for the load_bboxes module."""

from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr
from pytest import DATA_PATHS

from movement import MovementDataset
from movement.io import load_bboxes


@pytest.fixture(
    params=[
        "VIA_multiple-crabs_5-frames_labels.csv",
    ]
)
def via_tracks_file(request):
    """Return the file path for a VIA tracks .csv file."""
    return pytest.DATA_PATHS.get(request.param)


@pytest.fixture()
def valid_from_numpy_inputs():
    n_frames = 5
    n_individuals = 86
    n_space = 2
    individual_names_array = np.arange(n_individuals).reshape(-1, 1)

    rng = np.random.default_rng(seed=42)

    return {
        "position_array": rng.random((n_frames, n_individuals, n_space)),
        "shape_array": rng.random((n_frames, n_individuals, n_space)),
        "confidence_array": rng.random((n_frames, n_individuals)),
        "individual_names": [
            f"id_{id}" for id in individual_names_array.squeeze()
        ],
        "frame_array": np.arange(n_frames).reshape(-1, 1),
    }


def assert_dataset(dataset, file_path=None, expected_source_software=None):
    """Assert that the dataset is a proper xarray Dataset."""
    assert isinstance(dataset, xr.Dataset)

    # Expected variables are present and of right shape/type
    for var in ["position", "shape", "confidence"]:
        assert var in dataset.data_vars
        assert isinstance(dataset[var], xr.DataArray)
    assert dataset.position.ndim == 3
    assert dataset.shape.ndim == 3
    assert dataset.confidence.shape == dataset.position.shape[:-1]

    # Check the dims and coords
    DIM_NAMES = tuple(a for a in MovementDataset.dim_names if a != "keypoints")
    assert all([i in dataset.dims for i in DIM_NAMES])
    for d, dim in enumerate(DIM_NAMES[1:]):
        assert dataset.sizes[dim] == dataset.position.shape[d + 1]
        assert all([isinstance(s, str) for s in dataset.coords[dim].values])
    assert all([i in dataset.coords["space"] for i in ["x", "y"]])

    # Check the values?

    # Check the metadata attributes
    assert (
        dataset.source_file is None
        if file_path is None
        else dataset.source_file == file_path.as_posix()
    )
    assert (
        dataset.source_software is None
        if expected_source_software is None
        else dataset.source_software == expected_source_software
    )
    assert dataset.fps is None


def test_load_from_VIA_tracks_file(via_tracks_file):
    """Test that loading tracked bounding box data from
    a valid VIA tracks csv file returns a proper Dataset.
    """
    ds = load_bboxes.from_via_tracks_file(via_tracks_file)
    assert_dataset(ds, via_tracks_file, "VIA-tracks")


@pytest.mark.parametrize(
    "fps, expected_fps, expected_time_unit",
    [
        (None, None, "frames"),
        (-5, None, "frames"),
        (0, None, "frames"),
        (30, 30, "seconds"),
        (60.0, 60, "seconds"),
    ],
)
def test_fps_and_time_coords(fps, expected_fps, expected_time_unit):
    """Test that time coordinates are set according to the provided fps."""
    ds = load_bboxes.from_via_tracks_file(
        DATA_PATHS.get("VIA_multiple-crabs_5-frames_labels.csv"),
        fps=fps,
    )
    assert ds.time_unit == expected_time_unit

    if expected_fps is None:
        assert ds.fps is expected_fps
    else:
        assert ds.fps == expected_fps
        np.testing.assert_allclose(
            ds.coords["time"].data,
            np.arange(ds.sizes["time"], dtype=int) / ds.attrs["fps"],
        )


@pytest.mark.parametrize("source_software", ["Unknown", "VIA-tracks"])
@pytest.mark.parametrize("fps", [None, 30, 60.0])
def test_from_file_delegates_correctly(source_software, fps):
    """Test that the from_file() function delegates to the correct
    loader function according to the source_software.
    """
    software_to_loader = {
        "VIA-tracks": "movement.io.load_bboxes.from_via_tracks_file",
    }

    if source_software == "Unknown":
        with pytest.raises(ValueError, match="Unsupported source"):
            load_bboxes.from_file("some_file", source_software)
    else:
        with patch(software_to_loader[source_software]) as mock_loader:
            load_bboxes.from_file("some_file", source_software, fps)
            mock_loader.assert_called_with("some_file", fps)


@pytest.mark.parametrize("source_software", [None, "VIA-tracks"])
def test_from_numpy_valid(
    valid_from_numpy_inputs,
    source_software,
):
    """Test that loading bounding boxes trajectories from a multi-animal
    numpy array with valid parameters returns a proper Dataset.
    """
    ds = load_bboxes.from_numpy(
        **valid_from_numpy_inputs,
        fps=None,
        source_software=source_software,
    )
    assert_dataset(ds, expected_source_software=source_software)
