import numpy as np
import pytest

from movement.io import save_bboxes
from movement.io.save_bboxes import _write_single_via_row


@pytest.mark.parametrize(
    "confidence",
    [None, 0.5],
    ids=["without_confidence", "with_confidence"],
)
@pytest.mark.parametrize(
    "filename_prefix",
    [None, "test_video"],
    ids=["without_filename_prefix", "with_filename_prefix"],
)
@pytest.mark.parametrize(
    "all_frames_size",
    [None, 100],
    ids=["without_all_frames_size", "with_all_frames_size"],
)
@pytest.mark.parametrize(
    "max_digits",
    [5, 3],
    ids=["max_digits_5", "max_digits_3"],
)
def test_write_single_via_row(
    confidence,
    filename_prefix,
    all_frames_size,
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
    row = _write_single_via_row(
        frame,
        track_id,
        xy_coordinates,
        wh_values,
        max_digits,
        confidence,
        filename_prefix,
        all_frames_size,
    )

    # Compute expected values
    filename_prefix = f"{f'{filename_prefix}_' if filename_prefix else ''}"
    expected_filename = f"{filename_prefix}{frame:0{max_digits}d}.jpg"
    expected_file_size = all_frames_size if all_frames_size is not None else 0
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


def test_to_via_tracks_file_valid_dataset():
    """Test the VIA-tracks CSV file."""
    # Test different valid datasets, including with gaps
    pass


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
            tmp_path / "test.csv",
        )


def test_to_via_tracks_file_invalid_file_path():
    """Test that an invalid file path raises an error."""
    pass


def test_to_via_tracks_file_with_nans():
    """Test the VIA-tracks CSV file."""
    pass


def test_to_via_tracks_file_with_confidence():
    """Test the VIA-tracks CSV file."""
    pass
