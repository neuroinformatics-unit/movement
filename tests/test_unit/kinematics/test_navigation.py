# tests/test_unit/kinematics/test_navigation.py
import pytest
import xarray as xr

from movement.kinematics.navigation import (
    compute_forward_vector,
)


@pytest.mark.parametrize(
    "input_data, expected_error, expected_match_str, keypoints",
    [
        (
            "not_a_dataset",
            TypeError,
            "must be an xarray.DataArray",
            ["left_ear", "right_ear"],
        ),
        (
            "missing_dim_poses_dataset",
            ValueError,
            (
                r"Input data must have ['time', 'keypoints', 'space'] "
                r"as dimensions.\n"
                r"Input data must contain ['left_ear', 'right_ear'] in the "
                r"'keypoints' coordinates"
            ),
            ["left_ear", "right_ear"],
        ),
        (
            "missing_two_dims_bboxes_dataset",
            ValueError,
            (
                r"Input data must have ['time', 'keypoints', 'space'] "
                r"as dimensions.\n"
                r"Input data must contain ['left_ear', 'right_ear'] in the "
                r"'keypoints' coordinates"
            ),
            ["left_ear", "right_ear"],
        ),
        (
            "valid_poses_dataset",
            ValueError,
            r"Input data must contain ['left_ear', 'left_ear'] in the "
            r"'keypoints' coordinates",
            ["left_ear", "left_ear"],
        ),
    ],
)
def test_compute_forward_vector_with_invalid_input(
    input_data, expected_error, expected_match_str, keypoints, request
):
    with pytest.raises(expected_error, match=expected_match_str):
        data = request.getfixturevalue(input_data)
        if isinstance(data, xr.Dataset):
            data = data.position
        compute_forward_vector(data, keypoints[0], keypoints[1])


def test_compute_forward_vector_identical_keypoints(
    valid_data_array_for_forward_vector,
):
    data = valid_data_array_for_forward_vector
    with pytest.raises(ValueError, match="keypoints may not be identical"):
        compute_forward_vector(data, "left_ear", "left_ear")
