"""Test suite for the load_bboxes module."""

import ast
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pytest import DATA_PATHS

from movement import MovementDataset
from movement.io import load_bboxes


@pytest.fixture()
def via_tracks_file():
    """Return the file path for a VIA tracks .csv file."""
    via_sample_file_name = "VIA_multiple-crabs_5-frames_labels.csv"
    return pytest.DATA_PATHS.get(via_sample_file_name)


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


@pytest.fixture()
def df_input_via_tracks_small(via_tracks_file):
    """Return the first 3 rows of the VIA tracks .csv file as a dataframe."""
    df = pd.read_csv(via_tracks_file, sep=",", header=0)
    return df.loc[:2, :]


@pytest.fixture()
def df_input_via_tracks_small_with_confidence(df_input_via_tracks_small):
    """Return a dataframe with the first three rows of the VIA tracks .csv file
    and add confidence values to the bounding boxes.
    """
    df = update_attribute_column(
        df_input=df_input_via_tracks_small,
        attribute_column_name="region_attributes",
        dict_to_append={"confidence": "0.5"},
    )

    return df


@pytest.fixture()
def df_input_via_tracks_small_with_frame_number(df_input_via_tracks_small):
    """Return a dataframe with the first three rows of the VIA tracks .csv file
    and add frame number values to the bounding boxes.
    """
    df = update_attribute_column(
        df_input=df_input_via_tracks_small,
        attribute_column_name="file_attributes",
        dict_to_append={"frame": "1"},
    )

    return df


def update_attribute_column(df_input, attribute_column_name, dict_to_append):
    """Update an attributes column in the dataframe."""
    # copy the dataframe
    df = df_input.copy()

    # get the attributes column and convert to dict
    attributes_dicts = [ast.literal_eval(d) for d in df[attribute_column_name]]

    # update the dict
    for d in attributes_dicts:
        d.update(dict_to_append)

    # update the region_attributes column in the dataframe
    df[attribute_column_name] = [str(d) for d in attributes_dicts]
    return df


def assert_dataset(dataset, file_path=None, expected_source_software=None):
    """Assert that the dataset is a proper ``movement`` Dataset."""
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


@pytest.mark.parametrize("source_software", ["Unknown", "VIA-tracks"])
@pytest.mark.parametrize("fps", [None, 30, 60.0])
def test_from_file(source_software, fps):
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


def test_from_VIA_tracks_file(via_tracks_file):
    """Test that loading tracked bounding box data from
    a valid VIA tracks .csv file returns a proper Dataset.
    """
    ds = load_bboxes.from_via_tracks_file(via_tracks_file)
    assert_dataset(ds, via_tracks_file, "VIA-tracks")


@pytest.mark.parametrize("source_software", [None, "VIA-tracks"])
def test_from_numpy(
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


@pytest.mark.parametrize(
    "via_column_name, list_keys, cast_fn, expected_attribute_array",
    [
        (
            "file_attributes",
            ["clip"],
            int,
            np.array([123] * 3),  # .reshape(-1, 1),
        ),
        (
            "region_shape_attributes",
            ["name"],
            str,
            np.array(["rect"] * 3),  # .reshape(-1, 1),
        ),
        (
            "region_shape_attributes",
            ["x", "y"],
            float,
            np.array(
                [
                    [526.2366942646654, 393.280914246804],
                    [2565, 468],
                    [759.6484377108334, 136.60946673708338],
                ]
            ).reshape(-1, 2),
        ),
        (
            "region_shape_attributes",
            ["width", "height"],
            float,
            np.array([[46, 38], [41, 30], [29, 25]]).reshape(-1, 2),
        ),
        (
            "region_attributes",
            ["track"],
            int,
            np.array([71, 70, 69]),  # .reshape(-1, 1),
        ),
    ],
)
def test_via_attribute_column_to_numpy(
    df_input_via_tracks_small,
    via_column_name,
    list_keys,
    cast_fn,
    expected_attribute_array,
):
    """Test that the function correctly extracts the desired data from the VIA
    attributes.
    """
    attribute_array = load_bboxes._via_attribute_column_to_numpy(
        df=df_input_via_tracks_small,
        via_column_name=via_column_name,
        list_keys=list_keys,
        cast_fn=cast_fn,
    )

    assert np.array_equal(attribute_array, expected_attribute_array)


@pytest.mark.parametrize(
    "df_input, expected_array",
    [
        ("df_input_via_tracks_small", np.full((3,), np.nan)),
        (
            "df_input_via_tracks_small_with_confidence",
            np.array([0.5, 0.5, 0.5]),
        ),
    ],
)
def test_extract_confidence_from_via_tracks_df(
    df_input, expected_array, request
):
    """Test that the function correctly extracts the confidence values from
    the VIA dataframe.
    """
    df = request.getfixturevalue(df_input)
    confidence_array = load_bboxes._extract_confidence_from_via_tracks_df(df)

    assert np.array_equal(confidence_array, expected_array, equal_nan=True)


@pytest.mark.parametrize(
    "df_input, expected_array",
    [
        (
            "df_input_via_tracks_small",
            np.ones((3,)),
        ),  # extract from filename
        (
            "df_input_via_tracks_small_with_frame_number",
            np.array([1, 1, 1]),
        ),  # extract from file_attributes
    ],
)
def test_extract_frame_number_from_via_tracks_df(
    df_input, expected_array, request
):
    """Test that the function correctly extracts the frame number values from
    the VIA dataframe.
    """
    df = request.getfixturevalue(df_input)
    frame_array = load_bboxes._extract_frame_number_from_via_tracks_df(df)

    assert np.array_equal(frame_array, expected_array)


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
