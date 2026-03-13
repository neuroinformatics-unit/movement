"""Test suite for the load_bboxes module."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from movement.io import load_bboxes
from movement.validators.datasets import ValidBboxesInputs
from movement.validators.files import ValidVIATracksCSV


@pytest.fixture()
def get_expected_attributes_dict():
    """Define a factory of expected attributes dictionaries."""

    def _get_expected_attributes_dict(via_file_name: str) -> dict:
        """Return the expected attributes dictionary for the first 3 rows of
        the input VIA file.
        """
        attributes_dict_per_test_file = {}
        # add expected attributes for the first 3 rows of
        # VIA_single-crab_MOCA-crab-1.csv
        attributes_dict_per_test_file["VIA_single-crab_MOCA-crab-1.csv"] = {
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
        }
        # add expected attributes for the first 3 rows of
        # "VIA_multiple-crabs_5-frames_labels.csv"
        attributes_dict_per_test_file[
            "VIA_multiple-crabs_5-frames_labels.csv"
        ] = {
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
        }
        # return ValueError if the filename is not defined
        if via_file_name not in attributes_dict_per_test_file:
            raise ValueError(
                f"Attributes dict not defined for filename '{via_file_name}''."
            )
        return attributes_dict_per_test_file[via_file_name]

    return _get_expected_attributes_dict


@pytest.fixture()
def create_df_input_via_tracks():
    """Define a factory of dataframes for testing."""

    def _create_df_input_via_tracks(
        via_file_path: Path,
        small: bool = False,
        attribute_column_additions: dict[str, list[dict]] | None = None,
    ) -> pd.DataFrame:
        """Return an optionally modified dataframe that results from
        reading the input VIA tracks .csv filepath.

        If small is True, only the first 3 rows are returned.

        If attribute_column_additions is not None, the dataframe is modified
        to include the data under the specified attribute column.

        The variable attribute_column_additions is a dictionary mapping the
        name of the attribute column to a list of dictionaries to append to
        that column.
        """
        # read the VIA tracks .csv file as a dataframe
        df = pd.read_csv(via_file_path, sep=",", header=0)
        # optionally return the first 3 rows only
        if small:
            df = df.loc[:2, :]
        # optionally modify the dataframe to include data
        # under an attribute column
        if attribute_column_additions is None:
            return df
        else:
            return update_attribute_column(
                df_input=df,
                attribute_column_additions=attribute_column_additions,
            )

    return _create_df_input_via_tracks


@pytest.fixture()
def via_multiple_crabs_gap_id_1(tmp_path):
    """Return a filepath to a modified VIA tracks .csv file with
    the annotations for id=1 removed for frames 1, 2 and 3.
    """
    filepath = pytest.DATA_PATHS.get("VIA_multiple-crabs_5-frames_labels.csv")

    # Drop annotations for id=1 in frames 1, 2 and 3
    df = pd.read_csv(filepath)
    filename_prefix = "04.09.2023-04-Right_RE_testframe_0000000"
    for frame_number in range(1, 4):
        df = df[
            ~(
                (df["region_attributes"] == '{"track":"1"}')
                & (df["filename"] == f"{filename_prefix}{frame_number}.png")
            )
        ]

    # Save the modified dataframe to a new file
    filepath = tmp_path / "VIA_multiple-crabs_5-frames_labels_with_gap.csv"
    df.to_csv(filepath, index=False)

    return filepath


def update_attribute_column(
    df_input: pd.DataFrame,
    attribute_column_additions: dict[str, list[dict]],
):
    """Update an attributes column in the dataframe."""
    # copy the dataframe
    df = df_input.copy()
    # update each attribute column in the dataframe
    for attribute_column_name in attribute_column_additions:
        # get the list of dicts to append to the column
        list_dicts_to_append = attribute_column_additions[
            attribute_column_name
        ]
        # get the column to update, and convert it to a list of dicts
        # (one dict per row)
        attributes_dicts = [json.loads(d) for d in df[attribute_column_name]]
        # update each dict in the list
        # (if we only have one dict to append, append it to all rows)
        if len(list_dicts_to_append) == 1:
            for d in attributes_dicts:
                d.update(list_dicts_to_append[0])
        else:
            for d, dict_to_append in zip(
                attributes_dicts, list_dicts_to_append, strict=True
            ):
                d.update(dict_to_append)
        # update the relevant column in the dataframe and format
        # back to string
        df[attribute_column_name] = [str(d) for d in attributes_dicts]
    return df


@pytest.fixture()
def create_valid_from_numpy_inputs(rng):
    """Define a factory of valid inputs to "from_numpy" function."""
    n_frames = 5
    n_space = 2
    n_individuals = 86
    individual_names_array = np.arange(n_individuals).reshape(-1, 1)
    first_frame_number = 1  # should match sample file

    def _create_valid_from_numpy_inputs(with_frame_array=False):
        """Return a dictionary of valid inputs to the `from_numpy` function."""
        required_inputs = {
            "position_array": rng.random((n_frames, n_space, n_individuals)),
            "shape_array": rng.random((n_frames, n_space, n_individuals)),
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


def assert_time_coordinates(ds, fps, start_frame=None, frame_array=None):
    """Assert that the time coordinates are as expected, depending on
    fps value and start_frame or time_array. start_frame takes precedence
    over frame_array if both are provided.
    """
    # scale time coordinates with 1/fps if provided
    scale = 1 / fps if fps else 1
    # build frame array from start_frame if provided
    if start_frame is not None:
        frame_array = np.array(
            range(start_frame, len(ds.coords["time"].data) + start_frame)
        )
    elif frame_array is None:
        raise ValueError(
            "Either start_frame or frame_array must be provided."
            "start_frame takes precedence over frame_array if "
            "both are provided."
        )
    # assert numpy array of time coordinates
    np.testing.assert_allclose(
        ds.coords["time"].data, np.array([f * scale for f in frame_array])
    )


@pytest.mark.filterwarnings("ignore:.*is deprecated:DeprecationWarning")
@pytest.mark.parametrize("source_software", ["Unknown", "VIA-tracks"])
@pytest.mark.parametrize("fps", [None, 30, 60.0])
@pytest.mark.parametrize("use_frame_numbers_from_file", [True, False])
@pytest.mark.parametrize("frame_regexp", [None, r"frame_(\d+)"])
def test_from_file(
    source_software, fps, use_frame_numbers_from_file, frame_regexp
):
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
                frame_regexp=frame_regexp,
            )
    else:
        with patch(software_to_loader[source_software]) as mock_loader:
            load_bboxes.from_file(
                "some_file",
                source_software,
                fps,
                use_frame_numbers_from_file=use_frame_numbers_from_file,
                frame_regexp=frame_regexp,
            )
            mock_loader.assert_called_with(
                "some_file",
                fps,
                use_frame_numbers_from_file=use_frame_numbers_from_file,
                frame_regexp=frame_regexp,
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
@pytest.mark.parametrize("frame_regexp", [None, r"(00\d*)\.\w+$"])
def test_from_via_tracks_file(
    via_file_path, fps, use_frame_numbers_from_file, frame_regexp, helpers
):
    """Test that loading tracked bounding box data from
    a valid VIA tracks .csv file returns a proper Dataset.
    """
    kwargs = {
        "file": via_file_path,
        "fps": fps,
        "use_frame_numbers_from_file": use_frame_numbers_from_file,
        **({"frame_regexp": frame_regexp} if frame_regexp is not None else {}),
    }
    ds = load_bboxes.from_via_tracks_file(**kwargs)
    expected_values = {
        "vars_dims": {"position": 3, "shape": 3, "confidence": 2},
        "dim_names": ValidBboxesInputs.DIM_NAMES,
        "source_software": "VIA-tracks",
        "fps": fps,
        "file_path": via_file_path,
    }
    helpers.assert_valid_dataset(ds, expected_values)


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
    helpers,
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
    expected_values = {
        "vars_dims": {"position": 3, "shape": 3, "confidence": 2},
        "dim_names": ValidBboxesInputs.DIM_NAMES,
        "source_software": source_software,
        "fps": fps,
    }
    helpers.assert_valid_dataset(ds, expected_values)
    # check time coordinates are as expected
    start_frame = (
        from_numpy_inputs["frame_array"][0, 0]
        if "frame_array" in from_numpy_inputs
        else 0
    )
    assert_time_coordinates(ds, fps, start_frame)


@pytest.mark.parametrize(
    "input_data",
    [
        # Case 1: all IDs are defined for all frames
        # expected df is just sorted
        {
            "ids": np.array([1, 2, 1, 2]),
            "frame_numbers": np.array([1, 1, 0, 0]),
            "x": np.array([10.0, 20.0, 30.0, 40.0]),
            "y": np.array([1.0, 2.0, 3.0, 4.0]),
            "w": np.array([5.0, 5.0, 5.0, 5.0]),
            "h": np.array([5.0, 5.0, 5.0, 5.0]),
            "confidence": np.array([1.0, 1.0, 1.0, 1.0]),
        },
        # Case 2: ID=2 is not defined for frame 1
        # (nan should be added)
        {
            "ids": np.array([1, 1, 2]),
            "frame_numbers": np.array([0, 1, 0]),
            "x": np.array([10.0, 20.0, 30.0]),
            "y": np.array([1.0, 2.0, 3.0]),
            "w": np.array([5.0, 5.0, 5.0]),
            "h": np.array([5.0, 5.0, 5.0]),
            "confidence": np.array([1.0, 1.0, 1.0]),
        },
    ],
)
def test_numpy_arrays_from_valid_via_object(input_data):
    """Test all arrays extracted from a valid VIA file object."""
    mock_via = Mock()
    for key, val in input_data.items():
        setattr(mock_via, key, val)

    result = load_bboxes._numpy_arrays_from_valid_via_object(mock_via)

    # Parse input data
    x, y, w, h, ids, frames = (
        input_data[k] for k in ["x", "y", "w", "h", "ids", "frame_numbers"]
    )

    # Compute expected arrays
    unique_ids = np.unique(ids)
    unique_frames = np.unique(frames)
    n_individuals, n_frames = len(unique_ids), len(unique_frames)

    expected_position = np.full((n_frames, 2, n_individuals), np.nan)
    expected_shape = np.full((n_frames, 2, n_individuals), np.nan)
    expected_confidence = np.full((n_frames, n_individuals), np.nan)
    for i in range(len(x)):
        fi = np.searchsorted(unique_frames, frames[i])
        ii = np.searchsorted(unique_ids, ids[i])
        expected_position[fi, 0, ii] = x[i] + w[i] / 2
        expected_position[fi, 1, ii] = y[i] + h[i] / 2
        expected_shape[fi, 0, ii] = w[i]
        expected_shape[fi, 1, ii] = h[i]
        expected_confidence[fi, ii] = input_data["confidence"][i]

    np.testing.assert_array_equal(
        result["ID_array"], unique_ids.reshape(-1, 1)
    )
    np.testing.assert_array_equal(
        result["frame_array"], unique_frames.reshape(-1, 1)
    )
    np.testing.assert_allclose(
        result["position_array"], expected_position, equal_nan=True
    )
    np.testing.assert_allclose(
        result["shape_array"], expected_shape, equal_nan=True
    )
    np.testing.assert_allclose(
        result["confidence_array"], expected_confidence, equal_nan=True
    )


@pytest.mark.filterwarnings(
    "ignore:.*Setting fps to None.:UserWarning",
)
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
        assert getattr(ds, "fps", None) == expected_fps
    else:
        assert not hasattr(ds, "fps")
    # check loading frame numbers from file
    if use_frame_numbers_from_file:
        assert_time_coordinates(
            ds, expected_fps, frame_array=frame_array_from_file
        )
    else:
        assert_time_coordinates(ds, expected_fps, start_frame=0)


@pytest.mark.parametrize(
    "via_file_path",
    [
        pytest.DATA_PATHS.get("VIA_multiple-crabs_5-frames_labels.csv"),
        pytest.DATA_PATHS.get("VIA_single-crab_MOCA-crab-1.csv"),
    ],
)
def test_position_array_from_valid_via_object(via_file_path):
    """Test the position arrays extracted from a valid VIA file object."""
    # Extract numpy arrays from VIA tracks .csv file
    via_file_object = ValidVIATracksCSV(via_file_path)
    bboxes_arrays = load_bboxes._numpy_arrays_from_valid_via_object(
        via_file_object
    )

    # Get pre-parsed arrays from validator
    x = via_file_object.x
    y = via_file_object.y
    w = via_file_object.w
    h = via_file_object.h
    ids = via_file_object.ids
    frames = via_file_object.frame_numbers

    unique_ids = np.unique(ids)
    unique_frames = np.unique(frames)

    # Build expected position array independently
    expected_position = np.full(
        (len(unique_frames), 2, len(unique_ids)), np.nan
    )  # (frame, space, individual)
    # loop through observations
    for obs_i in range(len(x)):
        fi = np.searchsorted(unique_frames, frames[obs_i])
        ii = np.searchsorted(unique_ids, ids[obs_i])
        expected_position[fi, 0, ii] = x[obs_i] + w[obs_i] / 2
        expected_position[fi, 1, ii] = y[obs_i] + h[obs_i] / 2

    assert np.allclose(
        bboxes_arrays["position_array"],
        expected_position,
        equal_nan=True,
    )


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "via_file_path",
    [
        pytest.DATA_PATHS.get("VIA_multiple-crabs_5-frames_labels.csv"),
        # multiple individuals present in all 5 frames
        pytest.DATA_PATHS.get("VIA_single-crab_MOCA-crab-1.csv"),
        # single individual present in 35 non-consecutive frames
    ],
)
def test_benchmark_from_via_tracks_file(via_file_path, benchmark):
    """Benchmark the loading of a VIA tracks .csv file."""
    benchmark(load_bboxes.from_via_tracks_file, via_file_path)


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "via_file_path",
    [
        pytest.DATA_PATHS.get("VIA_multiple-crabs_5-frames_labels.csv"),
        # multiple individuals present in all 5 frames
        pytest.DATA_PATHS.get("VIA_single-crab_MOCA-crab-1.csv"),
        # single individual present in 35 non-consecutive frames
    ],
)
def test_benchmark_df_from_valid_via_object(via_file_path, benchmark):
    """Benchmark the `_df_from_valid_via_object` function."""
    valid_via = ValidVIATracksCSV(via_file_path)
    benchmark(load_bboxes._df_from_valid_via_object, valid_via)
