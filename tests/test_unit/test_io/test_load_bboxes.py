"""Test suite for the load_bboxes module."""

import ast
import re
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from movement.io import load_bboxes
from movement.validators.datasets import ValidBboxesInputs


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
        attributes_dicts = [
            ast.literal_eval(d) for d in df[attribute_column_name]
        ]
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
        "file_path": via_file_path,
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
    "frame_regexp, error_type, log_message",
    [
        (
            r"*",
            re.error,
            "The provided regular expression for the frame numbers (*) "
            "could not be compiled. Please review its syntax.",
        ),
        (
            r"_(0\d*)_$",
            AttributeError,
            "00000.jpg (row 0): "
            r"The provided frame regexp (_(0\d*)_$) did not return any "
            "matches and a frame number could not be extracted from "
            "the filename.",
        ),
        (
            r"(0\d*\.\w+)$",
            ValueError,
            "00000.jpg (row 0): "
            "The frame number extracted from the filename "
            r"using the provided regexp ((0\d*\.\w+)$) "
            "could not be cast as an integer.",
        ),
    ],
)
def test_from_via_tracks_file_invalid_frame_regexp(
    frame_regexp, error_type, log_message
):
    """Test that loading tracked bounding box data from
    a valid VIA tracks .csv file with an invalid frame_regexp
    raises a ValueError.
    """
    input_file = pytest.DATA_PATHS.get("VIA_single-crab_MOCA-crab-1.csv")
    with pytest.raises(error_type) as excinfo:
        load_bboxes.from_via_tracks_file(
            input_file,
            use_frame_numbers_from_file=True,
            frame_regexp=frame_regexp,
        )

    assert str(excinfo.value) == log_message


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
    get_expected_attributes_dict,
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
    attributes_dict = get_expected_attributes_dict(
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
    if input_confidence_value:
        df = create_df_input_via_tracks(
            via_file_path,
            small=True,  # only get 3 rows
            attribute_column_additions={
                "region_attributes": [{"confidence": input_confidence_value}]
            },
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
def test_extract_frame_number_from_via_tracks_df_filenames(
    create_df_input_via_tracks,
    via_file_path,
    expected_frame_array,
):
    """Test that the function correctly extracts the frame number values from
    the images' filenames.
    """
    # create the dataframe with the frame number
    df = create_df_input_via_tracks(
        via_file_path,
        small=True,
    )
    # the VIA tracks .csv files have no frames defined under the
    # "file_attributes" so the frame numbers should be extracted
    # from the filenames
    assert not all("frame" in row for row in df["file_attributes"])
    # extract frame number from df
    frame_array = load_bboxes._extract_frame_number_from_via_tracks_df(df)
    assert np.array_equal(frame_array, expected_frame_array)


@pytest.mark.parametrize(
    "via_file_path, attribute_column_additions, expected_frame_array",
    [
        (
            pytest.DATA_PATHS.get("VIA_multiple-crabs_5-frames_labels.csv"),
            {"file_attributes": [{"frame": 222}]},
            np.ones(
                3,
            )
            * 222,
        ),
        (
            pytest.DATA_PATHS.get("VIA_single-crab_MOCA-crab-1.csv"),
            {
                "file_attributes": [
                    {"frame": 218},
                    {"frame": 219},
                    {"frame": 220},
                ]
            },
            np.array([218, 219, 220]),
        ),
    ],
)
def test_extract_frame_number_from_via_tracks_df_file_attributes(
    create_df_input_via_tracks,
    via_file_path,
    attribute_column_additions,
    expected_frame_array,
):
    """Test that the function correctly extracts the frame number values from
    the file attributes column.

    The frame number defined under the "file_attributes" column
    should take precedence over the frame numbers encoded in the filenames.
    """
    # Create the dataframe with the frame number stored in
    # the file_attributes column
    df = create_df_input_via_tracks(
        via_file_path,
        small=True,
        attribute_column_additions=attribute_column_additions,
    )
    # extract frame number from the dataframe
    # (should take precedence over the frame numbers in the filenames)
    frame_array = load_bboxes._extract_frame_number_from_via_tracks_df(df)
    assert np.array_equal(frame_array, expected_frame_array)


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
    "via_file_path, expected_n_frames, expected_n_individuals",
    [
        (
            pytest.DATA_PATHS.get("VIA_multiple-crabs_5-frames_labels.csv"),
            5,
            86,
        ),  # multiple crabs present in all 5 frames
        (
            pytest.DATA_PATHS.get("VIA_single-crab_MOCA-crab-1.csv"),
            35,
            1,
        ),  # single crab present in 35 non-consecutive frames
        (
            "via_multiple_crabs_gap_id_1",
            5,
            86,
        ),  # multiple crabs, all but id=1 are present in all 5 frames
    ],
)
def test_df_from_via_tracks_file(
    via_file_path, expected_n_frames, expected_n_individuals, request
):
    """Test that the `_df_from_via_tracks_file` helper function correctly
    reads the VIA tracks .csv file as a dataframe.
    """
    if via_file_path == "via_multiple_crabs_gap_id_1":
        via_file_path = request.getfixturevalue(via_file_path)

    # Read the VIA tracks .csv file as a dataframe
    df = load_bboxes._df_from_via_tracks_file(via_file_path)

    # Check dataframe
    assert isinstance(df, pd.DataFrame)
    assert len(df.frame_number.unique()) == expected_n_frames
    assert len(df.ID.unique()) == expected_n_individuals

    # Check all individuals are present in all frames (even if nan)
    assert df.shape[0] == len(df.ID.unique()) * expected_n_frames

    # Check that the dataframe has the expected columns
    assert list(df.columns) == [
        "ID",
        "frame_number",
        "x",
        "y",
        "w",
        "h",
        "confidence",
    ]

    # Check that the dataframe is sorted by frame_number and ID
    assert df.sort_values(["ID", "frame_number"]).equals(df)


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
        bboxes_arrays["position_array"],  # frames, xy, individuals
        np.stack(list_derived_centroids, axis=-1),
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
def test_benchmark_df_from_via_tracks_file(via_file_path, benchmark):
    """Benchmark the `_df_from_via_tracks_file` function."""
    benchmark(load_bboxes._df_from_via_tracks_file, via_file_path)
