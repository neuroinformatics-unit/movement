from unittest.mock import Mock, patch

import numpy as np
import pytest

from movement.io import save_bboxes
from movement.io.save_bboxes import (
    _map_individuals_to_track_ids,
    _write_single_row,
)


@pytest.fixture
def mock_csv_writer():
    """Return a mock CSV writer object."""
    # Mock csv writer object
    writer = Mock()
    # Add writerow method to mock object
    writer.writerow = Mock()
    return writer


@pytest.mark.parametrize(
    "confidence",
    [None, 0.5],
    ids=["without_confidence", "with_confidence"],
)
@pytest.mark.parametrize(
    "image_file_prefix",
    [None, "test_video"],
    ids=["without_filename_prefix", "with_filename_prefix"],
)
@pytest.mark.parametrize(
    "image_file_suffix",
    [None, "png"],
    ids=["without_image_file_suffix", "with_image_file_suffix"],
)
@pytest.mark.parametrize(
    "image_size",
    [None, 100],
    ids=["without_all_frames_size", "with_all_frames_size"],
)
@pytest.mark.parametrize(
    "max_digits",
    [5, 3],
    ids=["max_digits_5", "max_digits_3"],
)
def test_write_single_row(
    mock_csv_writer,
    confidence,
    image_file_prefix,
    image_file_suffix,
    image_size,
    max_digits,
):
    """Test writing a single row of the VIA-tracks CSV file."""
    # Fixed input values
    frame, track_id, xy_coordinates, wh_values = (
        1,
        0,
        np.array([100, 200]),
        np.array([50, 30]),
    )

    # Write single row of VIA-tracks CSV file
    with patch("csv.writer", return_value=mock_csv_writer):
        row = _write_single_row(
            mock_csv_writer,
            xy_coordinates,
            wh_values,
            confidence,
            track_id,
            frame,
            max_digits,
            image_file_prefix,
            image_file_suffix,
            image_size,
        )
        mock_csv_writer.writerow.assert_called_with(row)

    # Compute expected values
    image_file_prefix = (
        f"{f'{image_file_prefix}_' if image_file_prefix else ''}"
    )
    expected_filename = (
        f"{image_file_prefix}{frame:0{max_digits}d}.{image_file_suffix}"
    )
    expected_file_size = image_size if image_size is not None else 0
    expected_file_attributes = "{}"  # placeholder value
    expected_region_count = 0  # placeholder value
    expected_region_id = 0  # placeholder value
    expected_region_shape_attributes = {
        "name": "rect",
        "x": float(xy_coordinates[0] - wh_values[0] / 2),
        "y": float(xy_coordinates[1] - wh_values[1] / 2),
        "width": float(wh_values[0]),
        "height": float(wh_values[1]),
    }
    expected_region_attributes = (
        f'{{"track":"{int(track_id)}", "confidence":"{confidence}"}}'
        if confidence is not None
        else f'{{"track":"{int(track_id)}"}}'
    )

    # Check values are as expected
    assert row[0] == expected_filename
    assert row[1] == expected_file_size
    assert row[2] == expected_file_attributes
    assert row[3] == expected_region_count
    assert row[4] == expected_region_id
    assert row[5] == f"{expected_region_shape_attributes}"
    assert row[6] == f"{expected_region_attributes}"


@pytest.mark.parametrize(
    "list_individuals, expected_track_id",
    [
        (["id_1", "id_3", "id_2"], [1, 3, 2]),
        (["id_1", "id_2", "id_3"], [1, 2, 3]),
        (["id-1", "id-2", "id-3"], [1, 2, 3]),
        (["id1", "id2", "id3"], [1, 2, 3]),
        (["id101", "id2", "id333"], [101, 2, 333]),
        (["mouse_0_id1", "mouse_0_id2"], [1, 2]),
    ],
    ids=[
        "unsorted",
        "sorted",
        "underscores",
        "dashes",
        "multiple_digits",
        "middle_and_end_digits",
    ],
)
def test_map_individuals_to_track_ids_from_individuals_names(
    list_individuals, expected_track_id
):
    """Test the mapping individuals to track IDs if the track ID is
    extracted from the individuals' names.
    """
    # Map individuals to track IDs
    map_individual_to_track_id = _map_individuals_to_track_ids(
        list_individuals, extract_track_id_from_individuals=True
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
    ids=["sorted", "unsorted", "ignoring_digits"],
)
def test_map_individuals_to_track_ids_factorised(
    list_individuals, expected_track_id
):
    """Test the mapping individuals to track IDs if the track ID is
    factorised from the sorted individuals' names.
    """
    # Map individuals to track IDs
    map_individual_to_track_id = _map_individuals_to_track_ids(
        list_individuals, extract_track_id_from_individuals=False
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
        (["A", "B", "C", "D"], "Could not extract track ID from A."),
    ],
    ids=["id_clash_1", "id_clash_2", "individuals_without_digits"],
)
def test_map_individuals_to_track_ids_error(
    list_individuals, expected_error_message
):
    """Test that an error is raised if extracting track IDs from the
    individuals' names fails.
    """
    with pytest.raises(ValueError) as error:
        _map_individuals_to_track_ids(
            list_individuals,
            extract_track_id_from_individuals=True,
        )

    # Check that the error message is as expected
    assert str(error.value) == expected_error_message


@pytest.mark.parametrize(
    "valid_dataset",
    [
        "valid_bboxes_dataset",
        "valid_bboxes_dataset_in_seconds",
        "valid_bboxes_dataset_with_nan",
        # "valid_bboxes_dataset_with_gaps", -- TODO
    ],
)
@pytest.mark.parametrize(
    "extract_track_id_from_individuals",
    [True, False],
)
@pytest.mark.parametrize(
    "image_file_prefix",
    [None, "test_video"],
)
@pytest.mark.parametrize(
    "image_file_suffix",
    [None, ".png"],
)
def test_to_via_tracks_file_valid_dataset(
    valid_dataset,
    request,
    tmp_path,
    extract_track_id_from_individuals,
    image_file_prefix,
    image_file_suffix,
):
    """Test the VIA-tracks CSV file."""
    # TODO: Test different valid datasets, including those
    # with IDs that are not present in all frames
    save_bboxes.to_via_tracks_file(
        request.getfixturevalue(valid_dataset),
        tmp_path / "test_valid_dataset.csv",
        extract_track_id_from_individuals,
        image_file_prefix=image_file_prefix,
        image_file_suffix=image_file_suffix,
    )

    # TODO: Check values are as expected!
    # TODO:Check as many track IDs as individuals


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


def test_to_via_tracks_file_without_confidence():
    """Test exporting a VIA-tracks CSV file when the dataset has no
    confidence values.
    """
    pass
