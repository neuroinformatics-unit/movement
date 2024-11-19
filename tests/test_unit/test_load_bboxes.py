"""Test suite for the load_bboxes module."""

import ast
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from movement.io import load_bboxes
from movement.validators.datasets import ValidBboxesDataset


@pytest.fixture()
def get_attributes_dict():
    def _get_attributes_dict(via_file_name: str) -> dict:
        # Should match the first 3 rows of each file!
        attributes_dict_per_test_file = {
            "VIA_single-crab_MOCA-crab-1.csv": {
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
            "VIA_multiple-crabs_5-frames_labels.csv": {
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
        }
        if via_file_name not in attributes_dict_per_test_file:
            raise ValueError(
                f"Attributes dict not defined for filename '{via_file_name}''."
            )
        return attributes_dict_per_test_file[via_file_name]

    return _get_attributes_dict


@pytest.fixture()
def create_df_input_via_tracks():
    def _create_df_input_via_tracks(
        via_file_path: Path,
        small: bool = False,
        attribute_column_name: str | None = None,
        dict_to_append: dict | None = None,
    ) -> pd.DataFrame:
        """Return the dataframe that results from
        reading the input VIA tracks .csv filepath.
        """
        # read the VIA tracks .csv file as a dataframe
        df = pd.read_csv(via_file_path, sep=",", header=0)
        # optionally return the first 3 rows only
        if small:
            df = df.loc[:2, :]

        # optionally modify the dataframe to include the attribute column
        if attribute_column_name is None:
            return df
        else:
            return update_attribute_column(
                df_input=df,
                attribute_column_name=attribute_column_name,  # "region_attributes", /// "file_attributes",
                dict_to_append=dict_to_append,  # {"confidence": "0.5"}, /// {"frame": "1"},
            )

    return _create_df_input_via_tracks


@pytest.fixture()
def create_valid_from_numpy_inputs():
    n_frames = 5
    n_individuals = 86
    n_space = 2
    individual_names_array = np.arange(n_individuals).reshape(-1, 1)
    first_frame_number = 1  # should match sample file

    rng = np.random.default_rng(seed=42)

    def _create_valid_from_numpy_inputs(with_frame_array=False):
        required_inputs = {
            "position_array": rng.random((n_frames, n_individuals, n_space)),
            "shape_array": rng.random((n_frames, n_individuals, n_space)),
            "confidence_array": rng.random((n_frames, n_individuals)),
            "individual_names": [
                f"id_{id}" for id in individual_names_array.squeeze()
            ],
        }

        if with_frame_array:
            required_inputs["frame_array"] = np.arange(
                first_frame_number, first_frame_number + n_frames
            ).reshape(-1, 1)

        return required_inputs

    return _create_valid_from_numpy_inputs


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
    "via_file_path",
    [
        pytest.DATA_PATHS.get("VIA_multiple-crabs_5-frames_labels.csv"),
        pytest.DATA_PATHS.get("VIA_single-crab_MOCA-crab-1.csv"),
    ],
)
@pytest.mark.parametrize("fps", [None, 30, 60.0])
@pytest.mark.parametrize("use_frame_numbers_from_file", [True, False])
def test_from_via_tracks_file(
    via_file_path,
    fps,
    use_frame_numbers_from_file,
):
    """Test that loading tracked bounding box data from
    a valid VIA tracks .csv file returns a proper Dataset.
    """
    ds = load_bboxes.from_via_tracks_file(
        file_path=via_file_path,
        fps=fps,
        use_frame_numbers_from_file=use_frame_numbers_from_file,
    )
    assert_dataset(ds, via_file_path, "VIA-tracks", fps)


@pytest.mark.parametrize(
    "with_frame_array",
    [True, False],
)
@pytest.mark.parametrize("fps", [None, 30, 60.0])
@pytest.mark.parametrize("source_software", [None, "VIA-tracks"])
def test_from_numpy(
    create_valid_from_numpy_inputs,
    with_frame_array,
    fps,
    source_software,
):
    """Test that loading bounding boxes trajectories from the input
    numpy arrays returns a proper Dataset.
    """
    # get the input arrays
    from_numpy_inputs = create_valid_from_numpy_inputs(with_frame_array)

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
    "via_file_path",
    [
        pytest.DATA_PATHS.get("VIA_multiple-crabs_5-frames_labels.csv"),
        pytest.DATA_PATHS.get("VIA_single-crab_MOCA-crab-1.csv"),
    ],
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
    create_df_input_via_tracks,
    get_attributes_dict,
    via_file_path,
    via_column_name,
    list_keys,
    cast_fn,
):
    """Test that the function correctly extracts the desired data from the VIA
    attributes.
    """
    attribute_array = load_bboxes._via_attribute_column_to_numpy(
        df=create_df_input_via_tracks(
            via_file_path, small=True
        ),  # small=True to only get 3 rows
        via_column_name=via_column_name,
        list_keys=list_keys,
        cast_fn=cast_fn,
    )

    attributes_dict = get_attributes_dict(
        via_file_path.name
    )  # returns results for the first 3 rows
    expected_attribute_array = attributes_dict[via_column_name][
        "_".join(list_keys)
    ]
    assert np.array_equal(attribute_array, expected_attribute_array)


@pytest.mark.parametrize(
    "via_file_path",
    [
        pytest.DATA_PATHS.get("VIA_multiple-crabs_5-frames_labels.csv"),
        pytest.DATA_PATHS.get("VIA_single-crab_MOCA-crab-1.csv"),
    ],
)
@pytest.mark.parametrize(
    "input_confidence_value, expected_confidence_array",
    # we only check the first 3 rows of the files
    [
        (
            None,
            np.full((3,), np.nan),
        ),
        (
            0.5,
            np.array([0.5, 0.5, 0.5]),
        ),
    ],
)
def test_extract_confidence_from_via_tracks_df(
    create_df_input_via_tracks,
    via_file_path,
    input_confidence_value,
    expected_confidence_array,
):
    """Test that the function correctly extracts the confidence values from
    the VIA dataframe.

    A mock VIA dataframe is generated with all confidence values set to the
    input_confidence_value.
    """
    # None of the sample files includes a confidence column
    # so we add it to the dataframe here
    # ----> can I use a crab one from tracking output & replace
    # VIA_multiple-crabs_5-frames_labels?
    if input_confidence_value:
        df = create_df_input_via_tracks(
            via_file_path,
            small=True,  # only get 3 rows
            attribute_column_name="region_attributes",
            dict_to_append={"confidence": input_confidence_value},
        )
    else:
        df = create_df_input_via_tracks(via_file_path, small=True)

    confidence_array = load_bboxes._extract_confidence_from_via_tracks_df(df)

    assert np.array_equal(
        confidence_array, expected_confidence_array, equal_nan=True
    )


@pytest.mark.parametrize(
    "via_file_path, expected_frame_array",
    [
        (
            pytest.DATA_PATHS.get("VIA_multiple-crabs_5-frames_labels.csv"),
            np.ones((3,)),
        ),
        (
            pytest.DATA_PATHS.get("VIA_single-crab_MOCA-crab-1.csv"),
            np.array([0, 5, 10]),
        ),
    ],
)
def test_extract_frame_number_from_via_tracks_df(
    create_df_input_via_tracks,
    via_file_path,
    expected_frame_array,
):
    """Test that the function correctly extracts the frame number values from
    the VIA dataframe.
    """
    df = create_df_input_via_tracks(via_file_path, small=True)
    frame_array = load_bboxes._extract_frame_number_from_via_tracks_df(df)

    assert np.array_equal(frame_array, expected_frame_array)


@pytest.mark.parametrize(
    "via_file_path, frame_array_from_file",
    [
        (
            pytest.DATA_PATHS.get("VIA_multiple-crabs_5-frames_labels.csv"),
            np.array(range(1, 6)),
        ),
        (
            pytest.DATA_PATHS.get("VIA_single-crab_MOCA-crab-1.csv"),
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
    via_file_path,
    frame_array_from_file,
    fps,
    expected_fps,
    expected_time_unit,
    use_frame_numbers_from_file,
):
    """Test that fps conversion is as expected, and time coordinates are set
    according to the input "fps" and the "use_frame_numbers_from_file"
    parameters.
    """
    # load dataset with inputs
    ds = load_bboxes.from_via_tracks_file(
        via_file_path,
        fps=fps,
        use_frame_numbers_from_file=use_frame_numbers_from_file,
    )

    # check time unit
    assert ds.time_unit == expected_time_unit

    # check fps is as expected
    if bool(expected_fps):
        assert ds.fps == expected_fps
    else:
        assert ds.fps is None

    # check loading frame numbers from file
    if use_frame_numbers_from_file:
        assert_time_coordinates(
            ds, expected_fps, frame_array=frame_array_from_file
        )
    else:
        assert_time_coordinates(ds, expected_fps, start_frame=0)


@pytest.mark.parametrize(
    "via_file_path, expected_n_frames, expected_n_individuals",
    [
        (
            pytest.DATA_PATHS.get("VIA_multiple-crabs_5-frames_labels.csv"),
            5,
            86,
        ),
        (pytest.DATA_PATHS.get("VIA_single-crab_MOCA-crab-1.csv"), 35, 1),
    ],
)
def test_df_from_via_tracks_file(
    via_file_path, expected_n_frames, expected_n_individuals
):
    """Test that the `_df_from_via_tracks_file` helper function correctly
    reads the VIA tracks .csv file as a dataframe.
    """
    df = load_bboxes._df_from_via_tracks_file(
        via_file_path,
    )

    assert isinstance(df, pd.DataFrame)
    assert len(df.frame_number.unique()) == expected_n_frames
    assert len(df.ID.unique()) == expected_n_individuals
    assert (
        df.shape[0] == len(df.ID.unique()) * expected_n_frames
    )  # all individuals are present in all frames (even if nan)
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
    "via_file_path",
    [
        pytest.DATA_PATHS.get("VIA_multiple-crabs_5-frames_labels.csv"),
        pytest.DATA_PATHS.get("VIA_single-crab_MOCA-crab-1.csv"),
    ],
)
def test_position_numpy_array_from_via_tracks_file(via_file_path):
    """Test the extracted position array from the VIA tracks .csv file
    represents the centroid of the bbox.
    """
    # Extract numpy arrays from VIA tracks .csv file
    bboxes_arrays = load_bboxes._numpy_arrays_from_via_tracks_file(
        via_file_path
    )

    # Read VIA tracks .csv file as a dataframe
    df = load_bboxes._df_from_via_tracks_file(via_file_path)

    # Compute centroid positions from the dataframe
    # (go through in the same order as ID array)
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
