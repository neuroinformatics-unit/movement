"""Test suite for the load_bboxes module."""

import ast
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from movement.io import load_bboxes
from movement.validators.datasets import ValidBboxesDataset


# ---------
@pytest.fixture()
def single_animal_via_tracks_file():
    """Return the file path for a VIA tracks .csv file with a single animal."""
    via_sample_file_name = "VIA_single-crab_MOCA-crab-1.csv"
    return pytest.DATA_PATHS.get(via_sample_file_name)


@pytest.fixture()
def multi_animal_via_tracks_file():
    """Return the file path for a VIA tracks .csv file with multiple
    animals.
    """
    via_sample_file_name = "VIA_multiple-crabs_5-frames_labels.csv"
    return pytest.DATA_PATHS.get(via_sample_file_name)


# ---------


@pytest.fixture()
def df_input_via_tracks_small(request):
    """Return the first 3 rows of the dataframe that results from
    reading the input VIA tracks .csv filepath.
    """
    df = pd.read_csv(request.param, sep=",", header=0)
    return df.loc[:2, :]


@pytest.fixture()
def df_input_via_tracks_small_with_confidence(request):
    """Return the first 3 rows of the dataframe that results from
    reading the input VIA tracks .csv filepath and add some
    confidence values.
    """
    df = pd.read_csv(request.param, sep=",", header=0).loc[:2, :]

    return update_attribute_column(
        df_input=df,
        attribute_column_name="region_attributes",
        dict_to_append={"confidence": "0.5"},
    )


@pytest.fixture()
def df_input_via_tracks_small_with_frame_number(request):
    """Return the first 3 rows of the dataframe that results from
    reading the input VIA tracks .csv filepath and add some
    frame numbers.
    """
    df = pd.read_csv(request.param, sep=",", header=0).loc[:2, :]

    return update_attribute_column(
        df_input=df,
        attribute_column_name="file_attributes",
        dict_to_append={"frame": "1"},
    )


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


# -------------
# def df_input_multi_animal_via_tracks_small(multi_animal_via_tracks_file):
#     """Return the first 3 rows of the VIA tracks .csv file as a dataframe."""
#     df = pd.read_csv(multi_animal_via_tracks_file, sep=",", header=0)
#     return df.loc[:2, :]


# @pytest.fixture()
# def df_input_multi_animal_via_tracks_small_w_conf(
#     df_input_multi_animal_via_tracks_small,
# ):
#     """Return a dataframe with the first three rows of the
#     VIA tracks .csv file
#     and add confidence values to the bounding boxes.
#     """
#     df = update_attribute_column(
#         df_input=df_input_multi_animal_via_tracks_small,
#         attribute_column_name="region_attributes",
#         dict_to_append={"confidence": "0.5"},
#     )

#     return df


# @pytest.fixture()
# def df_input_via_tracks_small_with_frame_number(
#     df_input_multi_animal_via_tracks_small,
# ):
#     """Return a dataframe with the first three rows of the VIA tracks .csv
# file
#     and add frame number values to the bounding boxes.
#     """
# df = update_attribute_column(
#     df_input=df_input_multi_animal_via_tracks_small,
#     attribute_column_name="file_attributes",
#     dict_to_append={"frame": "1"},
# )

#     return df
# ---------


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


def assert_time_coordinates(ds, fps, start_frame=None, frame_array=None):
    """Assert that the time coordinates are as expected, depending on
    fps value and start_frame or time_array. start_frame takes precedence
    over frame_array if both are provided.
    """
    # scale time coordinates with 1/fps if provided
    scale = 1 / fps if fps else 1

    # build frame array if not provided
    if start_frame is not None:
        frame_array = np.array(
            range(start_frame, len(ds.coords["time"].data) + start_frame)
        )
    # elif not frame_array:
    #     frame_array = np.arange(1, len(ds.coords["time"].data) + 1)

    # assert numpy array of time coordinates
    np.testing.assert_allclose(
        ds.coords["time"].data, np.array([f * scale for f in frame_array])
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


@pytest.mark.parametrize(
    "via_tracks_file_str",
    ["multi_animal_via_tracks_file", "single_animal_via_tracks_file"],
)
@pytest.mark.parametrize("fps", [None, 30, 60.0])
@pytest.mark.parametrize("use_frame_numbers_from_file", [True, False])
def test_from_via_tracks_file(
    via_tracks_file_str, fps, use_frame_numbers_from_file, request
):
    """Test that loading tracked bounding box data from
    a valid VIA tracks .csv file returns a proper Dataset.
    """
    # get the input file
    via_tracks_file = request.getfixturevalue(via_tracks_file_str)

    # run general dataset checks
    ds = load_bboxes.from_via_tracks_file(
        file_path=via_tracks_file,
        fps=fps,
        use_frame_numbers_from_file=False,
    )
    assert_dataset(ds, via_tracks_file, "VIA-tracks", fps)


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
    "df_input_via_tracks_small, expected_attributes_dict",
    [
        (  # multi animal VIA tracks file
            pytest.DATA_PATHS.get("VIA_multiple-crabs_5-frames_labels.csv"),
            {
                "region_shape_attributes": {
                    "name": np.array(["rect"] * 3),
                    "x_y": np.array(
                        [
                            [526.2366942646654, 393.280914246804],
                            [2565, 468],
                            [759.6484377108334, 136.60946673708338],
                        ]
                    ).reshape(-1, 2),
                    "width_height": np.array(
                        [[46, 38], [41, 30], [29, 25]]
                    ).reshape(-1, 2),
                },
                "region_attributes": {
                    "track": np.array([71, 70, 69]),
                },
            },
        ),
        (
            # single animal VIA tracks file
            pytest.DATA_PATHS.get("VIA_single-crab_MOCA-crab-1.csv"),
            {
                "region_shape_attributes": {
                    "name": np.array(["rect"] * 3),
                    "x_y": np.array(
                        [
                            [957.85, 296.708],
                            [919.149, 296.708],
                            [889.317, 303.158],
                        ]
                    ).reshape(-1, 2),
                    "width_height": np.array(
                        [
                            [320.09, 153.191],
                            [357.984, 158.835],
                            [366.853, 162.867],
                        ]
                    ).reshape(-1, 2),
                },
                "region_attributes": {
                    "track": np.ones((3,)),
                },
            },
        ),
    ],
    indirect=["df_input_via_tracks_small"],
)
@pytest.mark.parametrize(
    "via_column_name, list_keys, cast_fn",
    [
        (
            "region_shape_attributes",
            ["name"],
            str,
        ),
        (
            "region_shape_attributes",
            ["x", "y"],
            float,
        ),
        (
            "region_shape_attributes",
            ["width", "height"],
            float,
        ),
        (
            "region_attributes",
            ["track"],
            int,
        ),
    ],
)
def test_via_attribute_column_to_numpy(
    df_input_via_tracks_small,
    expected_attributes_dict,
    via_column_name,
    list_keys,
    cast_fn,
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

    expected_attribute_array = expected_attributes_dict[via_column_name][
        "_".join(list_keys)
    ]
    assert np.array_equal(attribute_array, expected_attribute_array)


# TODO
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


# TODO
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
    "via_tracks_file, frame_array_from_file",
    [
        ("multi_animal_via_tracks_file", np.array(range(1, 6))),
        (
            "single_animal_via_tracks_file",
            np.array(list(range(0, 168, 5)) + [167]),
        ),
    ],
)
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
    frame_array_from_file,
    fps,
    expected_fps,
    expected_time_unit,
    use_frame_numbers_from_file,
    request,
):
    """Test that fps conversion is as expected, and time coordinates are set
    according to the input "fps" and the "use_frame_numbers_from_file"
    parameters.
    """
    via_tracks_file = request.getfixturevalue(via_tracks_file)

    # load dataset with inputs
    ds = load_bboxes.from_via_tracks_file(
        via_tracks_file,
        fps=fps,
        use_frame_numbers_from_file=use_frame_numbers_from_file,
    )

    # check time unit
    assert ds.time_unit == expected_time_unit

    # check fps is as expected
    if expected_fps is None:
        assert ds.fps is expected_fps
    else:
        assert ds.fps == expected_fps

    # check loading frame numbers from file
    if use_frame_numbers_from_file:
        assert_time_coordinates(
            ds, expected_fps, frame_array=frame_array_from_file
        )
    else:
        assert_time_coordinates(ds, expected_fps, start_frame=0)


@pytest.mark.parametrize(
    "via_tracks_file, expected_n_frames, expected_n_individuals",
    [
        ("multi_animal_via_tracks_file", 5, 86),
        ("single_animal_via_tracks_file", 35, 1),
    ],
)
def test_df_from_via_tracks_file(
    via_tracks_file, expected_n_frames, expected_n_individuals, request
):
    """Test that the `_df_from_via_tracks_file` helper function correctly
    reads the VIA tracks .csv file as a dataframe.
    """
    via_tracks_file = request.getfixturevalue(via_tracks_file)

    df = load_bboxes._df_from_via_tracks_file(
        file_path=via_tracks_file,
    )

    assert isinstance(df, pd.DataFrame)
    assert len(df.frame_number.unique()) == expected_n_frames
    assert len(df.ID.unique()) == expected_n_individuals
    assert (
        df.shape[0] == len(df.ID.unique()) * expected_n_frames
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


@pytest.mark.parametrize(
    "via_tracks_file",
    ["multi_animal_via_tracks_file", "single_animal_via_tracks_file"],
)
def test_position_numpy_array_from_via_tracks_file(via_tracks_file, request):
    """Test the extracted position array from the VIA tracks .csv file
    represents the centroid of the bbox.
    """
    via_tracks_file = request.getfixturevalue(via_tracks_file)

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
