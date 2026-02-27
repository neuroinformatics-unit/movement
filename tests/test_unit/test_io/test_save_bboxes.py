import json
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from movement.io import load_bboxes, save_bboxes
from movement.io.save_bboxes import (
    _compute_individuals_to_track_ids_map,
    _write_single_row,
)


@pytest.fixture
def mock_csv_writer():
    """Return a mock CSV writer object."""
    # Mock object
    writer = Mock()
    # Add writerow method to the mock object
    writer.writerow = Mock()
    return writer


@pytest.fixture
def valid_bboxes_dataset_min_frame_number_modified(valid_bboxes_dataset):
    """Return a valid bbboxes dataset with data for 10 frames,
    starting at frame number 333.

    `valid_bboxes_dataset` is a dataset with the time coordinate in
    frames and data for 10 frames.
    """
    return valid_bboxes_dataset.assign_coords(
        time=valid_bboxes_dataset.time + 333
    )


@pytest.fixture
def valid_bboxes_dataset_with_late_id0(valid_bboxes_dataset):
    """Return a valid bboxes dataset with id_0 starting at time index 3.

    `valid_bboxes_dataset` represents two individuals moving in uniform
    linear motion for 10 frames, with low confidence values and time in frames.
    """
    valid_bboxes_dataset.position.loc[
        {"individual": "id_0", "time": [0, 1, 2]}
    ] = np.nan
    return valid_bboxes_dataset


@pytest.fixture
def valid_bboxes_dataset_individuals_modified(valid_bboxes_dataset):
    """Return a valid bboxes dataset with individuals named "id_333" and
    "id_444".
    """
    valid_bboxes_dataset.assign_coords(individual=["id_333", "id_444"])
    return valid_bboxes_dataset


@pytest.fixture
def valid_bboxes_dataset_confidence_all_nans(valid_bboxes_dataset):
    """Return a valid bboxes dataset with all NaNs in
    the confidence array.
    """
    valid_bboxes_dataset["confidence"] = xr.DataArray(
        data=np.nan,
        dims=valid_bboxes_dataset.confidence.dims,
        coords=valid_bboxes_dataset.confidence.coords,
    )
    return valid_bboxes_dataset


@pytest.fixture
def valid_bboxes_dataset_confidence_some_nans(valid_bboxes_dataset):
    """Return a valid bboxes dataset with some NaNs in
    the confidence array.

    `valid_bboxes_dataset` represents two individuals moving in uniform
    linear motion for 10 frames, with time in frames. The confidence values
    for the first 3 frames for individual 0 are set to NaN.
    """
    # Set first 3 frames for individual 0 to NaN
    confidence_array = valid_bboxes_dataset.confidence.values
    confidence_array[:3, 0] = np.nan

    valid_bboxes_dataset["confidence"] = xr.DataArray(
        data=confidence_array,
        dims=valid_bboxes_dataset.confidence.dims,
        coords=valid_bboxes_dataset.confidence.coords,
    )
    return valid_bboxes_dataset


def _get_min_required_digits_in_ds(ds):
    """Return the minimum number of digits required to represent the
    largest frame number in the input dataset.
    """
    # Compute the maximum frame number
    max_frame_number = max(ds.time.values)
    if "seconds" in ds.time_unit:
        max_frame_number = int(max_frame_number * ds.fps)

    # Return the minimum number of digits required to represent the
    # largest frame number
    return len(str(max_frame_number))


@pytest.mark.parametrize(
    "valid_dataset",
    [
        "valid_bboxes_dataset",
        "valid_bboxes_dataset_in_seconds",
        "valid_bboxes_dataset_with_nan",  # nans in position array
        "valid_bboxes_dataset_with_late_id0",
    ],
)
def test_to_via_tracks_file_valid_dataset(
    valid_dataset,
    tmp_path,
    request,
):
    """Test the VIA tracks .csv file with different valid bboxes datasets."""
    # Save VIA tracks .csv file
    input_dataset = request.getfixturevalue(valid_dataset)
    output_path = tmp_path / "test_valid_dataset.csv"
    save_bboxes.to_via_tracks_file(input_dataset, output_path)

    # Check that the exported file is readable in movement
    if input_dataset.time_unit == "seconds":
        ds = load_bboxes.from_via_tracks_file(
            output_path, fps=input_dataset.fps
        )
    else:
        ds = load_bboxes.from_via_tracks_file(output_path)

    # Check the dataset matches the original one.
    # If the position or shape data arrays contain NaNs, remove those
    # data points from the original dataset before comparing (these bboxes
    # are skipped when writing the VIA tracks .csv file)
    null_position_or_shape = (
        input_dataset.position.isnull() | input_dataset.shape.isnull()
    )
    input_dataset.shape.values[null_position_or_shape] = np.nan
    input_dataset.position.values[null_position_or_shape] = np.nan
    input_dataset.confidence.values[np.any(null_position_or_shape, axis=1)] = (
        np.nan
    )
    xr.testing.assert_equal(ds, input_dataset)


@pytest.mark.parametrize(
    "image_file_prefix",
    [None, "test_video"],
)
@pytest.mark.parametrize(
    "image_file_suffix",
    [None, ".png", "png", ".jpg"],
)
def test_to_via_tracks_file_image_filename(
    valid_bboxes_dataset,
    image_file_prefix,
    image_file_suffix,
    tmp_path,
):
    """Test the VIA tracks .csv export with different image file prefixes and
    suffixes.
    """
    # Prepare kwargs
    kwargs = {"image_file_prefix": image_file_prefix}
    if image_file_suffix is not None:
        kwargs["image_file_suffix"] = image_file_suffix

    # Save VIA tracks .csv file
    output_path = tmp_path / "test_valid_dataset.csv"
    save_bboxes.to_via_tracks_file(
        valid_bboxes_dataset,
        output_path,
        **kwargs,
    )

    # Check image file prefix is as expected
    df = pd.read_csv(output_path)
    if image_file_prefix is not None:
        assert df["filename"].str.startswith(image_file_prefix).all()
    else:
        assert df["filename"].str.startswith("0").all()

    # Check image file suffix is as expected
    if image_file_suffix is not None:
        assert df["filename"].str.endswith(image_file_suffix).all()
    else:
        assert df["filename"].str.endswith(".png").all()


@pytest.mark.parametrize(
    "valid_dataset, expected_confidence_nan_count",
    [
        ("valid_bboxes_dataset", 0),
        # all bboxes should have a confidence value
        ("valid_bboxes_dataset_confidence_all_nans", 20),
        # some bboxes should have a confidence value
        ("valid_bboxes_dataset_confidence_some_nans", 3),
        # no bboxes should have a confidence value
    ],
)
def test_to_via_tracks_file_confidence(
    valid_dataset,
    expected_confidence_nan_count,
    tmp_path,
    request,
):
    """Test that the VIA tracks .csv file is as expected when the confidence
    array contains NaNs.
    """
    # Save VIA tracks .csv file
    input_dataset = request.getfixturevalue(valid_dataset)
    output_path = tmp_path / "test_valid_dataset.csv"
    save_bboxes.to_via_tracks_file(input_dataset, output_path)

    # Check that the input dataset has the expected number of NaNs in the
    # confidence array
    confidence_is_nan = input_dataset.confidence.isnull().values
    assert np.sum(confidence_is_nan) == expected_confidence_nan_count

    # Check that the confidence values in the exported file match the dataset
    df = pd.read_csv(output_path)
    df["region_attributes"] = [
        json.loads(el) for el in df["region_attributes"]
    ]

    # Check the "confidence" region attribute is present for
    # as many rows as there are non-NaN confidence values
    assert sum(
        ["confidence" in row for row in df["region_attributes"]]
    ) == np.sum(~confidence_is_nan)


@pytest.mark.parametrize(
    "valid_dataset",
    [
        "valid_bboxes_dataset",
        # individuals: "id_0", "id_1"
        "valid_bboxes_dataset_individuals_modified",
        # individuals: "id_333", "id_444"
    ],
)
@pytest.mark.parametrize(
    "track_ids_from_trailing_numbers",
    [True, False],
)
def test_to_via_tracks_file_track_ids_from_trailing_numbers(
    valid_dataset,
    track_ids_from_trailing_numbers,
    tmp_path,
    request,
):
    """Test that the VIA tracks .csv file is as expected when extracting
    track IDs from the individuals' names.
    """
    # Save VIA tracks .csv file
    output_path = tmp_path / "test_valid_dataset.csv"
    input_dataset = request.getfixturevalue(valid_dataset)
    save_bboxes.to_via_tracks_file(
        input_dataset,
        output_path,
        track_ids_from_trailing_numbers=track_ids_from_trailing_numbers,
    )

    # Check track ID in relation to individuals' names
    df = pd.read_csv(output_path)
    df["region_attributes"] = [
        json.loads(el) for el in df["region_attributes"]
    ]
    set_unique_track_ids = {
        int(row["track"]) for row in df["region_attributes"]
    }

    # Note: we check if the sets of IDs is as expected, regardless of the order
    if track_ids_from_trailing_numbers:
        assert set_unique_track_ids == {
            int(indiv.split("_")[1])
            for indiv in input_dataset.individual.values
        }
    else:
        assert set_unique_track_ids == {0, 1}


@pytest.mark.parametrize(
    "valid_dataset",
    [
        "valid_bboxes_dataset",
        "valid_bboxes_dataset_with_nan",
        "valid_bboxes_dataset_with_late_id0",
    ],
)
def test_to_via_tracks_file_region_count_and_id(
    valid_dataset, tmp_path, request
):
    """Test that the region count and region ID are as expected."""
    # Save VIA tracks .csv file
    output_path = tmp_path / "test_valid_dataset.csv"
    input_dataset = request.getfixturevalue(valid_dataset)
    save_bboxes.to_via_tracks_file(input_dataset, output_path)

    # Read output file as a dataframe
    df = pd.read_csv(output_path)

    # Check that the region count matches the number of annotations
    # per filename
    df_bboxes_count = df["filename"].value_counts(sort=False)
    map_filename_to_bboxes_count = dict(
        zip(df_bboxes_count.index, df_bboxes_count, strict=True)
    )
    assert all(
        df["region_count"].values
        == [map_filename_to_bboxes_count[fn] for fn in df["filename"]]
    )

    # Check that the region ID per filename ranges from 0 to the
    # number of annotations per filename
    assert all(
        np.all(
            df["region_id"].values[df["filename"] == fn]
            == np.array(range(map_filename_to_bboxes_count[fn]))
        )
        for fn in df["filename"]
    )


@pytest.mark.parametrize(
    "invalid_dataset, expected_exception",
    [
        ("not_a_dataset", TypeError),
        ("empty_dataset", ValueError),
        ("missing_var_bboxes_dataset", ValueError),
        ("missing_two_vars_bboxes_dataset", ValueError),
        ("missing_dim_bboxes_dataset", ValueError),
        ("missing_two_dims_bboxes_dataset", ValueError),
    ],
)
def test_to_via_tracks_file_invalid_dataset(
    invalid_dataset, expected_exception, request, tmp_path
):
    """Test that an invalid dataset raises an error."""
    with pytest.raises(expected_exception):
        save_bboxes.to_via_tracks_file(
            request.getfixturevalue(invalid_dataset),
            tmp_path / "test_invalid_dataset.csv",
        )


@pytest.mark.parametrize(
    "wrong_extension",
    [
        ".mp4",
        "",
    ],
)
def test_to_via_tracks_file_invalid_file_path(
    valid_bboxes_dataset, tmp_path, wrong_extension
):
    """Test that file with wrong extension raises an error."""
    with pytest.raises(ValueError):
        save_bboxes.to_via_tracks_file(
            valid_bboxes_dataset,
            tmp_path / f"test{wrong_extension}",
        )


@pytest.mark.parametrize(
    "frame_n_digits",
    [1, 100],
    ids=["1_digit", "100_digits"],
)
@pytest.mark.parametrize(
    "image_file_prefix, expected_prefix",
    [
        (None, ""),
        ("", ""),
        ("test_video", "test_video"),
        ("test_video_", "test_video_"),
    ],
    ids=["no_prefix", "empty_prefix", "prefix", "prefix_underscore"],
)
@pytest.mark.parametrize(
    "image_file_suffix, expected_suffix",
    [
        (".png", ".png"),
        ("png", ".png"),
        (".jpg", ".jpg"),
    ],
    ids=["png_extension", "png_no_dot", "jpg_extension"],
)
def test_get_image_filename_template(
    frame_n_digits,
    image_file_prefix,
    expected_prefix,
    image_file_suffix,
    expected_suffix,
):
    """Test that the image filename template is as expected."""
    expected_image_filename = (
        f"{expected_prefix}{{:0{frame_n_digits}d}}{expected_suffix}"
    )
    assert (
        save_bboxes._get_image_filename_template(
            frame_n_digits=frame_n_digits,
            image_file_prefix=image_file_prefix,
            image_file_suffix=image_file_suffix,
        )
        == expected_image_filename
    )


@pytest.mark.parametrize(
    "valid_dataset_str,",
    [
        ("valid_bboxes_dataset"),
        ("valid_bboxes_dataset_in_seconds"),
        ("valid_bboxes_dataset_min_frame_number_modified"),
    ],
    ids=["min_2_digits", "min_2_digits_in_seconds", "min_3_digits"],
)
@pytest.mark.parametrize(
    "frame_n_digits",
    [None, 7],
    ids=["auto", "user"],
)
def test_get_min_required_digits_in_ds(
    valid_dataset_str,
    frame_n_digits,
    request,
):
    """Test that the number of digits to represent the frame number is
    computed as expected.
    """
    ds = request.getfixturevalue(valid_dataset_str)
    min_required_digits = _get_min_required_digits_in_ds(ds)

    # Compute expected number of digits in output
    if frame_n_digits is None:
        expected_out_digits = min_required_digits + 1
    else:
        expected_out_digits = frame_n_digits

    # Check the number of digits to use in the output is as expected
    assert (
        save_bboxes._check_frame_required_digits(
            ds=ds, frame_n_digits=frame_n_digits
        )
        == expected_out_digits
    )


@pytest.mark.parametrize(
    "valid_dataset_str, requested_n_digits",
    [
        ("valid_bboxes_dataset", 0),
        ("valid_bboxes_dataset_min_frame_number_modified", 2),
    ],
    ids=["min_2_digits", "min_3_digits"],
)
def test_get_min_required_digits_in_ds_error(
    valid_dataset_str, requested_n_digits, request
):
    """Test that an error is raised if the requested number of digits is
    not enough to represent all the frame numbers.
    """
    ds = request.getfixturevalue(valid_dataset_str)
    min_required_digits = _get_min_required_digits_in_ds(ds)

    with pytest.raises(ValueError) as error:
        save_bboxes._check_frame_required_digits(
            ds=ds, frame_n_digits=requested_n_digits
        )

    assert str(error.value) == (
        "The requested number of digits cannot be used to represent all the "
        f"frame numbers. Got {requested_n_digits}, but the maximum frame "
        f"number has {min_required_digits} digits."
    )


@pytest.mark.parametrize(
    "list_individuals, expected_track_id",
    [
        (["id1", "id2", "id3"], [1, 2, 3]),
        (["id1", "id3", "id2"], [1, 3, 2]),
        (["id-1", "id-2", "id-3"], [1, 2, 3]),
        (["id_1", "id_2", "id_3"], [1, 2, 3]),
        (["id101", "id2", "id333"], [101, 2, 333]),
        (["mouse_0_id1", "mouse_0_id2"], [1, 2]),
        (["mouse_1abc", "mouse_2abc"], [1, 2]),
    ],
    ids=[
        "sorted",
        "unsorted",
        "dashes",
        "underscores",
        "multiple_digits",
        "middle_and_end_digits",
        "non_digits_after_trailing_numbers",
    ],
)
def test_individuals_to_track_ids_map_from_individuals_names(
    list_individuals, expected_track_id
):
    """Test the mapping individuals to track IDs if the track ID is
    extracted from the individuals' names.
    """
    # Map individuals to track IDs
    map_individual_to_track_id = _compute_individuals_to_track_ids_map(
        list_individuals, track_ids_from_trailing_numbers=True
    )

    # Check values are as expected
    assert [
        map_individual_to_track_id[individual]
        for individual in list_individuals
    ] == expected_track_id


@pytest.mark.parametrize(
    "list_individuals, expected_track_id",
    [
        (["A", "B", "C"], [0, 1, 2]),
        (["C", "B", "A"], [2, 1, 0]),
        (["id99", "id88", "id77"], [2, 1, 0]),
    ],
    ids=["sorted", "unsorted", "should_ignore_digits"],
)
def test_individuals_to_track_ids_map_factorised(
    list_individuals, expected_track_id
):
    """Test the mapping individuals to track IDs if the track ID is
    factorised from the sorted individuals' names.
    """
    # Map individuals to track IDs
    map_individual_to_track_id = _compute_individuals_to_track_ids_map(
        list_individuals, track_ids_from_trailing_numbers=False
    )

    # Check values are as expected
    assert [
        map_individual_to_track_id[individual]
        for individual in list_individuals
    ] == expected_track_id


@pytest.mark.parametrize(
    "list_individuals, expected_error_message",
    [
        (
            ["mouse_1_id0", "mouse_2_id0"],
            (
                "Could not extract a unique track ID for all individuals. "
                "Expected 2 unique track IDs, but got 1."
            ),
        ),
        (
            ["mouse_id1.0", "mouse_id2.0"],
            (
                "Could not extract a unique track ID for all individuals. "
                "Expected 2 unique track IDs, but got 1."
            ),
        ),
        (["A_1", "B_2", "C"], "Could not extract track ID from C."),
    ],
    ids=["id_clash_1", "id_clash_2", "individuals_without_digits"],
)
def test_individuals_to_track_ids_map_error(
    list_individuals, expected_error_message
):
    """Test that the appropriate error is raised if extracting track IDs
    from the individuals' names fails.
    """
    with pytest.raises(ValueError) as error:
        _compute_individuals_to_track_ids_map(
            list_individuals,
            track_ids_from_trailing_numbers=True,
        )

    # Check that the error message is as expected
    assert str(error.value) == expected_error_message


@pytest.mark.parametrize(
    "confidence",
    [None, 0.5],
    ids=["without_confidence", "with_confidence"],
)
@pytest.mark.parametrize(
    "image_size",
    [None, 100],
    ids=["without_image_size", "with_image_size"],
)
@pytest.mark.parametrize(
    "img_filename_template",
    ["{:05d}.png", "{:03d}.jpg", "frame_{:03d}.jpg"],
    ids=["png_extension", "jpg_extension", "frame_prefix"],
)
def test_write_single_row(
    mock_csv_writer,
    confidence,
    image_size,
    img_filename_template,
):
    """Test writing a single row of the VIA tracks .csv file."""
    # Fixed input values
    frame, track_id, region_count, region_id, xy_values, wh_values = (
        1,
        0,
        88,
        0,
        np.array([100, 200]),
        np.array([50, 30]),
    )

    # Write single row of VIA tracks .csv file
    with patch("csv.writer", return_value=mock_csv_writer):
        row = _write_single_row(
            writer=mock_csv_writer,
            xy_values=xy_values,
            wh_values=wh_values,
            confidence=confidence,
            track_id=track_id,
            region_count=region_count,
            region_id=region_id,
            img_filename=img_filename_template.format(frame),
            image_size=image_size,
        )
        mock_csv_writer.writerow.assert_called_with(row)

    # Compute expected region shape attributes
    expected_region_shape_attrs_dict = {
        "name": "rect",
        "x": float(xy_values[0] - wh_values[0] / 2),
        "y": float(xy_values[1] - wh_values[1] / 2),
        "width": float(wh_values[0]),
        "height": float(wh_values[1]),
    }
    expected_region_shape_attributes = json.dumps(
        expected_region_shape_attrs_dict
    )

    # Compute expected region attributes
    expected_region_attributes_dict = {
        "track": int(track_id),
    }
    if confidence is not None:
        expected_region_attributes_dict["confidence"] = confidence

    expected_region_attributes = json.dumps(expected_region_attributes_dict)

    # Check values are as expected
    assert row[0] == img_filename_template.format(frame)
    assert row[1] == (image_size if image_size is not None else 0)
    assert row[2] == '{"shot": 0}'  # placeholder value
    assert row[3] == region_count
    assert row[4] == region_id
    assert row[5] == expected_region_shape_attributes
    assert row[6] == expected_region_attributes


def test_number_of_quotes_in_via_tracks_csv_file(
    valid_bboxes_dataset, tmp_path
):
    """Test the literal string for two lines of the VIA tracks .csv file.

    This is to verify that the quotes in the output VIA tracks .csv file are
    as expected. Without the required double quotes, the file won't be
    importable in the VIA annotation tool.

    The VIA tracks .csv file format has:
    - dictionary-like items wrapped around single double-quotes (")
    - keys in these dictionaries wrapped around double double-quotes ("")

    See an example of the VIA tracks .csv file format at:
    https://www.robots.ox.ac.uk/~vgg/software/via/docs/face_track_annotation.html
    """
    # Save VIA tracks .csv file
    output_path = tmp_path / "test_valid_dataset.csv"
    save_bboxes.to_via_tracks_file(valid_bboxes_dataset, output_path)

    # Read text file
    with open(output_path) as file:
        lines = file.readlines()

    # Check a line with bbox id_0
    assert lines[1] == (
        "00.png,"  # filename
        "0,"  # filesize
        '"{""shot"": 0}",'  # file attributes
        "2,"  # region_count
        "0,"  # region_id
        '"{""name"": ""rect"", '  # region shape attributes
        '""x"": -30.0, ""y"": -20.0, ""width"": 60.0, ""height"": 40.0}",'
        '"{""track"": 0, ""confidence"": 0.9}"\n'  # region attributes
    )

    # Check a line with bbox id_1
    assert lines[-1] == (
        "09.png,"  # filename
        "0,"  # filesize
        '"{""shot"": 0}",'  # file attributes
        "2,"  # region_count
        "1,"  # region_id
        '"{""name"": ""rect"", '  # region shape attributes
        '""x"": -21.0, ""y"": -29.0, ""width"": 60.0, ""height"": 40.0}",'
        '"{""track"": 1, ""confidence"": 0.9}"\n'  # region attributes
    )


@pytest.mark.parametrize(
    "via_file_path",
    [
        pytest.DATA_PATHS.get("VIA_multiple-crabs_5-frames_labels.csv"),
        pytest.DATA_PATHS.get("VIA_single-crab_MOCA-crab-1.csv"),
    ],
)
def test_to_via_tracks_file_is_recoverable(via_file_path, tmp_path):
    """Test that an exported VIA tracks .csv file can be loaded back into
    the a dataset that matches the original one.
    """
    # Load a bboxes dataset from a VIA tracks .csv file
    original_ds = load_bboxes.from_via_tracks_file(
        via_file_path, use_frame_numbers_from_file=True
    )

    # Export the dataset
    output_path = tmp_path / "test_via_file.csv"
    save_bboxes.to_via_tracks_file(
        original_ds,
        output_path,
        track_ids_from_trailing_numbers=True,
    )

    # Load the exported file
    recovered_ds = load_bboxes.from_via_tracks_file(
        output_path, use_frame_numbers_from_file=True
    )

    # Compare the original and recovered datasets
    xr.testing.assert_equal(original_ds, recovered_ds)
