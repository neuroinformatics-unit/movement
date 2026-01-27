"""Test suite for the load_bboxes module."""

import json
import re
from pathlib import Path
from unittest.mock import patch

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


def test_parsed_df_from_file(tmp_path):
    """Test the parsed dataframe is as expected."""
    # Create minimal VIA tracks CSV with one row
    header = (
        "filename,file_size,file_attributes,region_count,"
        "region_id,region_shape_attributes,region_attributes\n"
    )
    row = (
        "frame_001.png,"
        "12345,"
        '"{""frame"":42}",'
        "1,"
        "0,"
        '"{""name"":""rect"",""x"":10,""y"":20,""width"":30,""height"":40}",'
        '"{""track"":""1"",""confidence"":0.9}"'
    )
    file_path = tmp_path / "test_via.csv"
    file_path.write_text(header + row)

    # Ensure it is a valid VIA tracks .csv file
    via_file = ValidVIATracksCSV(file_path)

    # Compute parsed dataframe
    df = load_bboxes._parsed_df_from_valid_file_object(via_file.path)

    # Check column names
    assert df.columns.tolist() == [
        "ID",
        "frame_number",
        "x",
        "y",
        "w",
        "h",
        "confidence",
    ]

    # Check column types
    assert df["ID"].dtype == int
    assert df["frame_number"].dtype == int
    assert all(
        df[col].dtype == np.float32
        for col in [
            "x",
            "y",
            "w",
            "h",
            "confidence",
        ]
    )

    # Check string columns are removed
    assert all(
        x not in df.columns.tolist()
        for x in [
            "region_shape_attributes",
            "region_attributes",
            "file_attributes",
        ]
    )

    # Check values match strings above
    assert len(df) == 1
    assert df["ID"].iloc[0] == 1
    assert df["frame_number"].iloc[0] == 42
    assert df["x"].iloc[0] == pytest.approx(10.0)
    assert df["y"].iloc[0] == pytest.approx(20.0)
    assert df["w"].iloc[0] == pytest.approx(30.0)
    assert df["h"].iloc[0] == pytest.approx(40.0)
    assert df["confidence"].iloc[0] == pytest.approx(0.9)


@pytest.mark.parametrize(
    "region_attributes, expected_confidence",
    [
        ('"{""track"":""1""}"', np.nan),  # no confidence key
        ('"{""track"":""1"", ""confidence"":0.75}"', 0.75),  # with confidence
    ],
    ids=["no_confidence", "with_confidence"],
)
def test_parsed_df_from_file_confidence(
    region_attributes,
    expected_confidence,
    tmp_path,
):
    """Test that _parsed_df_from_file returns NaN for confidence
    when not defined, and the actual value when defined.
    """
    # Define a minimal VIA tracks .csv file with one row
    header = (
        "filename,file_size,file_attributes,region_count,"
        "region_id,region_shape_attributes,region_attributes\n"
    )
    row = (
        "frame_001.png,"
        "12345,"
        '"{""frame"":1}",'
        "1,"
        "0,"
        '"{""name"":""rect"",""x"":10,""y"":20,""width"":30,""height"":40}",'
        f"{region_attributes}"
    )
    file_path = tmp_path / "test_via_one_row.csv"
    file_path.write_text(header + row)

    # Ensure it is a valid VIA tracks .csv file
    via_file = ValidVIATracksCSV(file_path)

    # Compute parsed dataframe
    df = load_bboxes._parsed_df_from_valid_file_object(via_file.path)

    # Check confidence value in df are as expected
    if np.isnan(expected_confidence):
        assert all(np.isnan(df["confidence"].values))
    else:
        assert all(df["confidence"].values == expected_confidence)


@pytest.mark.parametrize(
    "filename, file_attributes, expected_frame_number",
    [
        ("any_filename.png", '"{""frame"": 42}"', 42),
        ("frame_0275.png", '"{""foo"": 123}"', 275),
        ("0275.png", '"{""foo"": 123}"', 275),
    ],
)
def test_parsed_df_from_file_frame_number(
    filename, file_attributes, expected_frame_number, tmp_path
):
    """Test frame number extraction from input VIA tracks .csv file.

    We test the case where the frame number is defined in the filename,
    and the case where the frame number is defined in the file_attributes.
    """
    # Define a 1-row csv VIA tracks file
    header = (
        "filename,file_size,file_attributes,region_count,"
        "region_id,region_shape_attributes,region_attributes\n"
    )
    row = (
        f"{filename},"
        "12345,"
        f"{file_attributes},"
        "1,"
        "0,"
        '"{""name"":""rect"",""x"":10,""y"":20,""width"":30,""height"":40}",'
        '"{""track"":""1""}"'
    )
    file_path = tmp_path / "test_via_one_row.csv"
    file_path.write_text(header + row)

    # Ensure it is a valid VIA tracks .csv file
    via_file = ValidVIATracksCSV(file_path)

    # Compute parsed dataframe
    df = load_bboxes._parsed_df_from_valid_file_object(via_file.path)

    # Check frame number is as expected
    assert all(df["frame_number"] == expected_frame_number)


@pytest.mark.parametrize(
    "input_df, expected_df",
    [
        # Case 1: all IDs are defined for all frames
        # expected_df is just sorted (no nan rows added)
        (
            pd.DataFrame(
                {
                    "ID": [1, 2, 1, 2],
                    "frame_number": [1, 1, 0, 0],
                    "foo": [10, 20, 30, 40],
                }
            ),  # input
            pd.DataFrame(
                {
                    "ID": [1, 1, 2, 2],
                    "frame_number": [0, 1, 0, 1],
                    "foo": [30, 10, 40, 20],
                }
            ),  # expected
        ),
        # Case 2: ID=2 is not defined for frame 1
        # expected_df has nan rows added and sorted
        (
            pd.DataFrame(
                {
                    "ID": [1, 1, 2],
                    "frame_number": [0, 1, 0],
                    "foo": [10.0, 20.0, 30.0],
                }
            ),
            pd.DataFrame(
                {
                    "ID": [1, 1, 2, 2],
                    "frame_number": [0, 1, 0, 1],
                    "foo": [10.0, 20.0, 30.0, np.nan],
                }
            ),
        ),
    ],
)
def test_fill_in_missing_rows(input_df, expected_df):
    """Test sorting and gap-filling of ID/frame combinations."""
    df = load_bboxes._fill_in_missing_rows_and_sort(input_df)
    pd.testing.assert_frame_equal(df, expected_df)


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
    bboxes_arrays = load_bboxes._numpy_arrays_from_valid_file_object(
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
    valid_via = ValidVIATracksCSV(via_file_path)
    benchmark(load_bboxes._parsed_df_from_valid_file_object, valid_via)
