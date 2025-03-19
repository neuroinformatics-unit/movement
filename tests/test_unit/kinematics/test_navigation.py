import numpy as np  # Add for NaN checks
import pytest
import xarray as xr

from movement.kinematics.navigation import (
    compute_forward_vector,
    compute_forward_vector_angle,
    compute_head_direction_vector,
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
                r"Input data must contain \['time'\] as dimensions\.\n"
                r"Input data must contain \['left_ear', 'right_ear'\] in the "
                r"'keypoints' coordinates"
            ),
            ["left_ear", "right_ear"],
        ),
        (
            "missing_two_dims_bboxes_dataset",
            ValueError,
            (
                r"Input data must contain \['time', 'keypoints', 'space'\] as "
                r"dimensions\.\n"
                r"Input data must contain \['left_ear', 'right_ear'\] in the "
                r"'keypoints' coordinates"
            ),
            ["left_ear", "right_ear"],
        ),
        (
            "valid_poses_dataset",
            ValueError,
            (
                r"Input data must contain \['left_ear', 'left_ear'\] in the "
                r"'keypoints' coordinates\."
            ),
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


def test_compute_forward_vector(valid_data_array_for_forward_vector):
    """Test forward vector computation with valid input."""
    forward_vector = compute_forward_vector(
        valid_data_array_for_forward_vector,
        "left_ear",
        "right_ear",
        camera_view="bottom_up",
    )
    known_vectors = np.array([[[0, -1]], [[1, 0]], [[0, 1]], [[-1, 0]]])
    assert isinstance(forward_vector, xr.DataArray)
    assert np.equal(forward_vector.values, known_vectors).all()


@pytest.mark.parametrize("nan_time", [1, 2])  # Expand to 2 tests
def test_nan_behavior_forward_vector(
    valid_data_array_for_forward_vector_with_nan, nan_time
):
    """Test forward vector computation with NaNs at different times."""
    data = valid_data_array_for_forward_vector_with_nan.copy()
    # Adjust NaN position dynamically based on nan_time
    data = data.where((data.time != nan_time) | (data.keypoints != "left_ear"))
    forward_vector = compute_forward_vector(data, "left_ear", "right_ear")
    for preserved_coord in ["time", "space", "individuals"]:
        assert np.all(forward_vector[preserved_coord] == data[preserved_coord])
    assert set(forward_vector["space"].values) == {"x", "y"}
    nan_values = forward_vector.sel(time=nan_time)
    assert np.isnan(nan_values).all()
    assert not np.isnan(
        forward_vector.sel(time=[t for t in data.time if t != nan_time])
    ).any()


@pytest.mark.parametrize("camera_view", ["top_down", "bottom_up"])
def test_compute_head_direction_vector(
    valid_data_array_for_forward_vector, camera_view
):
    data = valid_data_array_for_forward_vector
    result = compute_head_direction_vector(
        data, "left_ear", "right_ear", camera_view=camera_view
    )
    assert isinstance(result, xr.DataArray)
    assert "space" in result.dims
    assert "keypoints" not in result.dims


@pytest.mark.parametrize("in_degrees", [True, False])
def test_compute_forward_vector_angle(
    valid_data_array_for_forward_vector, in_degrees
):
    data = valid_data_array_for_forward_vector
    result = compute_forward_vector_angle(
        data, "left_ear", "right_ear", in_degrees=in_degrees
    )
    assert isinstance(result, xr.DataArray)
    assert "space" not in result.dims
    assert "keypoints" not in result.dims
