"""Test suite for the load_bboxes module."""

import ast
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from movement.io import load_bboxes
from movement.validators.datasets import ValidBboxesDataset


@pytest.fixture()
def via_tracks_file():
    """Return the file path for a VIA tracks .csv file."""
    via_sample_file_name = "VIA_multiple-crabs_5-frames_labels.csv"
    return pytest.DATA_PATHS.get(via_sample_file_name)


@pytest.fixture()
def valid_from_numpy_inputs_required_arrays():
    """Return a dictionary with valid numpy arrays for the `from_numpy()`
    loader, excluding the optional `frame_array`.
    """
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
    }


@pytest.fixture()
def valid_from_numpy_inputs_all_arrays(
    valid_from_numpy_inputs_required_arrays,
):
    """Return a dictionary with valid numpy arrays for the from_numpy() loader,
    including a `frame_array` that ranges from frame 1 to 5.
    """
    n_frames = valid_from_numpy_inputs_required_arrays["position_array"].shape[
        0
    ]
    first_frame_number = 1  # should match sample file

    valid_from_numpy_inputs_required_arrays["frame_array"] = np.arange(
        first_frame_number, first_frame_number + n_frames
    ).reshape(-1, 1)

    return valid_from_numpy_inputs_required_arrays


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


def assert_dataset(
    dataset, file_path=None, expected_source_software=None, expected_fps=None
):
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
    DIM_NAMES = ValidBboxesDataset.DIM_NAMES
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
    assert (
        dataset.fps is None
        if expected_fps is None
        else dataset.fps == expected_fps
    )


def assert_time_coordinates(ds, fps, start_frame):
    """Assert that the time coordinates are as expected, depending on
    fps value and start_frame.
    """
    # scale time coordinates with 1/fps if provided
    scale = 1 / fps if fps else 1

    # assert numpy array of time coordinates
    np.testing.assert_allclose(
        ds.coords["time"].data,
        np.array(
            [
                f * scale
                for f in range(
                    start_frame, len(ds.coords["time"].data) + start_frame
                )
            ]
        ),
    )


@pytest.mark.parametrize("source_software", ["Unknown", "VIA-tracks"])
@pytest.mark.parametrize("fps", [None, 30, 60.0])
@pytest.mark.parametrize("use_frame_numbers_from_file", [True, False])
def test_from_file(source_software, fps, use_frame_numbers_from_file):
    """Test that the from_file() function delegates to the correct
    loader function according to the source_software.
    """
    software_to_loader = {
        "VIA-tracks": "movement.io.load_bboxes.from_via_tracks_file",
    }

    if source_software == "Unknown":
        with pytest.raises(ValueError, match="Unsupported source"):
            load_bboxes.from_file(
                "some_file",
                source_software,
                fps,
                use_frame_numbers_from_file=use_frame_numbers_from_file,
            )
    else:
        with patch(software_to_loader[source_software]) as mock_loader:
            load_bboxes.from_file(
                "some_file",
                source_software,
                fps,
                use_frame_numbers_from_file=use_frame_numbers_from_file,
            )
            mock_loader.assert_called_with(
                "some_file",
                fps,
                use_frame_numbers_from_file=use_frame_numbers_from_file,
            )


@pytest.mark.parametrize("fps", [None, 30, 60.0])
@pytest.mark.parametrize("use_frame_numbers_from_file", [True, False])
def test_from_via_tracks_file(
    via_tracks_file, fps, use_frame_numbers_from_file
):
    """Test that loading tracked bounding box data from
    a valid VIA tracks .csv file returns a proper Dataset
    and that the time coordinates are as expected.
    """
    # run general dataset checks
    ds = load_bboxes.from_via_tracks_file(
        via_tracks_file, fps, use_frame_numbers_from_file
    )
    assert_dataset(ds, via_tracks_file, "VIA-tracks", fps)

    # check time coordinates are as expected
    # in sample VIA tracks .csv file frame numbers start from 1
    start_frame = 1 if use_frame_numbers_from_file else 0
    assert_time_coordinates(ds, fps, start_frame)


@pytest.mark.parametrize(
    "valid_from_numpy_inputs",
    [
        "valid_from_numpy_inputs_required_arrays",
        "valid_from_numpy_inputs_all_arrays",
    ],
)
@pytest.mark.parametrize("fps", [None, 30, 60.0])
@pytest.mark.parametrize("source_software", [None, "VIA-tracks"])
def test_from_numpy(valid_from_numpy_inputs, fps, source_software, request):
    """Test that loading bounding boxes trajectories from the input
    numpy arrays returns a proper Dataset.
    """
    # get the input arrays
    from_numpy_inputs = request.getfixturevalue(valid_from_numpy_inputs)

    # run general dataset checks
    ds = load_bboxes.from_numpy(
        **from_numpy_inputs,
        fps=fps,
        source_software=source_software,
    )
    assert_dataset(
        ds, expected_source_software=source_software, expected_fps=fps
    )

    # check time coordinates are as expected
    if "frame_array" in from_numpy_inputs:
        start_frame = from_numpy_inputs["frame_array"][0, 0]
    else:
        start_frame = 0
    assert_time_coordinates(ds, fps, start_frame)


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
        (30, 30.0, "seconds"),
        (60.0, 60.0, "seconds"),
    ],
)
@pytest.mark.parametrize("use_frame_numbers_from_file", [True, False])
def test_fps_and_time_coords(
    via_tracks_file,
    fps,
    expected_fps,
    expected_time_unit,
    use_frame_numbers_from_file,
):
    """Test that fps conversion is as expected and time coordinates are set
    according to the expected fps.
    """
    ds = load_bboxes.from_via_tracks_file(
        via_tracks_file,
        fps=fps,
        use_frame_numbers_from_file=use_frame_numbers_from_file,
    )

    # load dataset with frame numbers from file
    ds_in_frames_from_file = load_bboxes.from_via_tracks_file(
        via_tracks_file,
        fps=None,
        use_frame_numbers_from_file=True,
    )

    # check time unit
    assert ds.time_unit == expected_time_unit

    # check fps is as expected
    if expected_fps is None:
        assert ds.fps is expected_fps
    else:
        assert ds.fps == expected_fps

    # check time coordinates
    if use_frame_numbers_from_file:
        start_frame = ds_in_frames_from_file.coords["time"].data[0]
    else:
        start_frame = 0
    assert_time_coordinates(ds, expected_fps, start_frame)


def test_df_from_via_tracks_file(via_tracks_file):
    """Test that the helper function correctly reads the VIA tracks .csv file
    as a dataframe.
    """
    df = load_bboxes._df_from_via_tracks_file(
        file_path=via_tracks_file,
    )

    assert isinstance(df, pd.DataFrame)
    assert len(df.frame_number.unique()) == 5
    assert (
        df.shape[0] == len(df.ID.unique()) * 5
    )  # all individuals in all frames (even if nan)
    assert list(df.columns) == [
        "ID",
        "frame_number",
        "x",
        "y",
        "w",
        "h",
        "confidence",
    ]


def test_position_numpy_array_from_via_tracks_file(via_tracks_file):
    """Test the extracted position array from the VIA tracks .csv file
    represents the centroid of the bbox.
    """
    # Extract numpy arrays from VIA tracks .csv file
    bboxes_arrays = load_bboxes._numpy_arrays_from_via_tracks_file(
        via_tracks_file
    )

    # Read VIA tracks .csv file as a dataframe
    df = load_bboxes._df_from_via_tracks_file(via_tracks_file)

    # Compute centroid positions from the dataframe
    # (go thru in the same order as ID array)
    list_derived_centroids = []
    for id in bboxes_arrays["ID_array"]:
        df_one_id = df[df["ID"] == id.item()]
        centroid_position = np.array(
            [df_one_id.x + df_one_id.w / 2, df_one_id.y + df_one_id.h / 2]
        ).T  # frames, xy
        list_derived_centroids.append(centroid_position)

    # Compare to extracted position array
    assert np.allclose(
        bboxes_arrays["position_array"],  # frames, individuals, xy
        np.stack(list_derived_centroids, axis=1),
    )
