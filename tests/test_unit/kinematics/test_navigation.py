import numpy as np
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
    known_vectors = np.array(
        [[[0, -1]], [[np.nan, np.nan]], [[0, 1]], [[-1, 0]]]
    )
    assert isinstance(forward_vector, xr.DataArray)
    assert np.allclose(forward_vector.values, known_vectors, equal_nan=True)


# Added fixtures from main
@pytest.fixture
def valid_data_array_for_forward_vector():
    """Return a position data array for an individual with 3 keypoints
    (left ear, right ear and nose), tracked for 4 frames, in x-y space.
    """
    time = [0, 1, 2, 3]
    individuals = ["id_0"]
    keypoints = ["left_ear", "right_ear", "nose"]
    space = ["x", "y"]
    ds = xr.DataArray(
        [
            [[[1, 0], [-1, 0], [0, -1]]],  # time 0
            [[[0, 1], [0, -1], [1, 0]]],  # time 1
            [[[-1, 0], [1, 0], [0, 1]]],  # time 2
            [[[0, -1], [0, 1], [-1, 0]]],  # time 3
        ],
        dims=["time", "individuals", "keypoints", "space"],
        coords={
            "time": time,
            "individuals": individuals,
            "keypoints": keypoints,
            "space": space,
        },
    )
    return ds


@pytest.fixture
def valid_data_array_for_forward_vector_with_nan(
    valid_data_array_for_forward_vector,
):
    """Return a position DataArray where position values are NaN for the
    ``left_ear`` keypoint at time ``1``.
    """
    nan_dataarray = valid_data_array_for_forward_vector.where(
        (valid_data_array_for_forward_vector.time != 1)
        | (valid_data_array_for_forward_vector.keypoints != "left_ear")
    )
    return nan_dataarray


@pytest.fixture
def spinning_on_the_spot():
    """Simulate data for an individual's head spinning on the spot.

    The left / right keypoints move in a circular motion counter-clockwise
    around the unit circle centred on the origin, always opposite each
    other.
    The left keypoint starts on the negative x-axis, and the motion is
    split into 8 time points of uniform rotation angles.
    """
    x_axis = np.array([1.0, 0.0])
    y_axis = np.array([0.0, 1.0])
    sqrt_2 = np.sqrt(2.0)
    data = np.zeros(shape=(8, 2, 2), dtype=float)
    data[:, :, 0] = np.array(
        [
            -x_axis,
            (-x_axis - y_axis) / sqrt_2,
            -y_axis,
            (x_axis - y_axis) / sqrt_2,
            x_axis,
            (x_axis + y_axis) / sqrt_2,
            y_axis,
            (-x_axis + y_axis) / sqrt_2,
        ]
    )
    data[:, :, 1] = -data[:, :, 0]
    return xr.DataArray(
        data=data,
        dims=["time", "space", "keypoints"],
        coords={"space": ["x", "y"], "keypoints": ["left", "right"]},
    )


def push_into_range(angle, lower=-np.pi, upper=np.pi):
    """Wrap angles into a specified range.

    Args:
        angle: The angle to wrap.
        lower: The lower bound of the range (default: -pi).
        upper: The upper bound of the range (default: pi).

    Returns:
        The angle wrapped into the range [lower, upper).

    """
    return ((angle - lower) % (upper - lower)) + lower


def test_nan_behavior_forward_vector(
    valid_data_array_for_forward_vector_with_nan,
):
    """Test that ``compute_forward_vector()`` generates the
    expected output for a valid input DataArray containing ``NaN``
    position values at a single time (``1``) and keypoint
    (``left_ear``).
    """
    nan_time = 1
    forward_vector = compute_forward_vector(
        valid_data_array_for_forward_vector_with_nan, "left_ear", "right_ear"
    )
    for preserved_coord in ["time", "space", "individuals"]:
        assert np.all(
            forward_vector[preserved_coord]
            == valid_data_array_for_forward_vector_with_nan[preserved_coord]
        )
    assert set(forward_vector["space"].values) == {"x", "y"}
    nan_values = forward_vector.sel(time=nan_time)
    assert nan_values.shape == (1, 2)
    assert np.isnan(nan_values).all(), (
        "NaN values not returned where expected!"
    )
    assert not np.isnan(
        forward_vector.sel(
            time=[
                t
                for t in valid_data_array_for_forward_vector_with_nan.time
                if t != nan_time
            ]
        )
    ).any()


@pytest.mark.parametrize(
    ["swap_left_right", "swap_camera_view"],
    [
        pytest.param(True, True, id="(TT) LR, Camera"),
        pytest.param(True, False, id="(TF) LR"),
        pytest.param(False, True, id="(FT) Camera"),
        pytest.param(False, False, id="(FF)"),
    ],
)
def test_antisymmetry_properties(
    push_into_range,
    spinning_on_the_spot,
    swap_left_right,
    swap_camera_view,
):
    r"""Test antisymmetry arises where expected.

    Reversing the right and left keypoints, or the camera position, has the
    effect of mapping angles to the "opposite side" of the unit circle.
    Explicitly;
    - :math:`\theta <= 0` is mapped to :math:`\theta + 180`,
    - :math:`\theta > 0` is mapped to :math:`\theta - 180`.

    In theory, the antisymmetry of ``angle_rotates`` should be covered by
    the underlying tests for ``compute_signed_angle_2d``, however we
    include this case here for additional checks in conjunction with other
    behaviour.
    """
    reference_vector = np.array([1.0, 0.0])
    left_keypoint = "left"
    right_keypoint = "right"
    args_to_function = {}
    if swap_left_right:
        args_to_function["left_keypoint"] = right_keypoint
        args_to_function["right_keypoint"] = left_keypoint
    else:
        args_to_function["left_keypoint"] = left_keypoint
        args_to_function["right_keypoint"] = right_keypoint
    if swap_camera_view:
        args_to_function["camera_view"] = "bottom_up"
    with_orientations_swapped = compute_forward_vector_angle(
        data=spinning_on_the_spot,
        reference_vector=reference_vector,
        **args_to_function,  # type: ignore[arg-type]
    )
    without_orientations_swapped = compute_forward_vector_angle(
        data=spinning_on_the_spot,
        left_keypoint=left_keypoint,
        right_keypoint=right_keypoint,
        reference_vector=reference_vector,
    )
    expected_orientations = without_orientations_swapped.copy(deep=True)
    if swap_left_right:
        expected_orientations = push_into_range(
            expected_orientations + np.pi, lower=-np.pi, upper=np.pi
        )
    if swap_camera_view:
        expected_orientations = push_into_range(
            expected_orientations + np.pi, lower=-np.pi, upper=np.pi
        )
    expected_orientations = push_into_range(expected_orientations)
    xr.testing.assert_allclose(
        with_orientations_swapped, expected_orientations
    )


def test_in_degrees_toggle(spinning_on_the_spot):
    """Test that angles can be returned in degrees or radians."""
    reference_vector = np.array([1.0, 0.0])
    left_keypoint = "left"
    right_keypoint = "right"
    in_radians = compute_forward_vector_angle(
        data=spinning_on_the_spot,
        left_keypoint=left_keypoint,
        right_keypoint=right_keypoint,
        reference_vector=reference_vector,
        in_degrees=False,
    )
    in_degrees = compute_forward_vector_angle(
        data=spinning_on_the_spot,
        left_keypoint=left_keypoint,
        right_keypoint=right_keypoint,
        reference_vector=reference_vector,
        in_degrees=True,
    )
    xr.testing.assert_allclose(in_degrees, np.rad2deg(in_radians))


@pytest.mark.parametrize(
    ["transformation"],
    [pytest.param("scale"), pytest.param("translation")],
)
def test_transformation_invariance(
    spinning_on_the_spot,
    transformation,
):
    """Test that certain transforms of the data have no effect on
    the relative angle computed.

    - Translations applied to both keypoints (even if the translation
    changes with time) should not affect the result, so long as both
    keypoints receive the same translation (at each timepoint).
    - Scaling the right to left keypoint vector should not produce a
    different angle.
    """
    left_keypoint = "left"
    right_keypoint = "right"
    reference_vector = np.array([1.0, 0.0])
    translated_data = spinning_on_the_spot.values.copy()
    n_time_pts = translated_data.shape[0]
    if transformation == "translation":
        translated_data += np.arange(n_time_pts).reshape(n_time_pts, 1, 1)
    elif transformation == "scale":
        translated_data[:, :, 0] *= np.arange(1, n_time_pts + 1).reshape(
            n_time_pts, 1
        )
    else:
        raise ValueError(f"Did not recognise case: {transformation}")
    translated_data = spinning_on_the_spot.copy(
        deep=True, data=translated_data
    )
    untranslated_output = compute_forward_vector_angle(
        spinning_on_the_spot,
        left_keypoint=left_keypoint,
        right_keypoint=right_keypoint,
        reference_vector=reference_vector,
    )
    translated_output = compute_forward_vector_angle(
        translated_data,
        left_keypoint=left_keypoint,
        right_keypoint=right_keypoint,
        reference_vector=reference_vector,
    )
    xr.testing.assert_allclose(untranslated_output, translated_output)


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
