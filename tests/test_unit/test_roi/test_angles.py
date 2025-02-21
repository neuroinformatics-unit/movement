import re
from typing import Any, Literal

import numpy as np
import pytest
import xarray as xr

from movement.roi import LineOfInterest
from movement.roi.base import BaseRegionOfInterest


@pytest.fixture()
def sample_position_array() -> xr.DataArray:
    """Return a simulated position array to test the egocentric angle.

    The data has time, space, and keypoints dimensions.

    The keypoints are left, right, midpt (midpoint), and wild.
    The midpt is the mean of the left and right keypoints; the wild keypoint
    may be anywhere in the plane (it is used to test the ``position_keypoint``
    argument).

    time 1:
        left @ (1.25, 0.), right @ (1., -0.25), wild @ (-0.25, -0.25)

        Fwd vector is (1, -1).
    time 2:
        left @ (-0.25, 0.5), right @ (-0.25, 0.25), wild @ (-0.5, 0.375)

        Fwd vector is (-1, 0).
    time 3:
        left @ (0.375, 0.375), right @ (0.5, 0.375), wild @ (1.25, 1.25)

        Fwd vector is (0, -1).
    time 4:
        left @ (1., -0.25), right @ (1.25, 0.), wild @ (-0.25, -0.25)

        This is time 1 but with left and right swapped.
        Fwd vector is (-1, 1).
    time 5:
        left @ (0.25, 0.5), right @ (0.375, 0.25), wild @ (0.5, 0.65)

        Fwd vector is (-2, -1).
        acos(2/sqrt(5)) is the expected angle for the midpt.
        acos(1/sqrt(5)) should be that for the wild point IF going to the
        boundary.
    """
    points = np.zeros(shape=(5, 2, 4))
    points[:, :, 0] = [
        [1.25, 0.0],
        [-0.25, 0.5],
        [0.375, 0.375],
        [1.0, -0.25],
        [0.25, 0.5],
    ]
    points[:, :, 1] = [
        [1.0, -0.25],
        [-0.25, 0.25],
        [0.5, 0.375],
        [1.25, 0.0],
        [0.375, 0.25],
    ]
    points[:, :, 3] = [
        [-0.25, 1.25],
        [-0.5, 0.375],
        [1.25, 1.25],
        [-0.25, -0.25],
        [0.5, 0.65],
    ]
    points[:, :, 2] = np.mean(points[:, :, 0:2], axis=2)
    return xr.DataArray(
        data=points,
        dims=["time", "space", "keypoints"],
        coords={
            "space": ["x", "y"],
            "keypoints": ["left", "right", "midpt", "wild"],
        },
    )


@pytest.mark.parametrize(
    ["region", "data", "fn_args", "expected_output", "which_method"],
    [
        pytest.param(
            "unit_square_with_hole",
            "sample_position_array",
            {
                "left_keypoint": "left",
                "right_keypoint": "right",
                "angle_rotates": "elephant to region",
            },
            ValueError("Unknown angle convention: elephant to region"),
            "compute_egocentric_angle",
            id="[E] Unknown angle convention",
        ),
        pytest.param(
            "unit_square_with_hole",
            "sample_position_array",
            {
                "position_keypoint": "midpt",
                "angle_rotates": "elephant to region",
            },
            ValueError("Unknown angle convention: elephant to region"),
            "compute_allocentric_angle",
            id="[A] Unknown angle convention",
        ),
        pytest.param(
            "unit_square_with_hole",
            "sample_position_array",
            {
                "left_keypoint": "left",
                "right_keypoint": "right",
            },
            np.array(
                [
                    0.0,
                    180.0,
                    0.0,
                    180.0,
                    np.rad2deg(np.arccos(2.0 / np.sqrt(5.0))),
                ]
            ),
            "compute_egocentric_angle",
            id="[E] Default args",
        ),
        pytest.param(
            "unit_square_with_hole",
            "sample_position_array",
            {
                "left_keypoint": "left",
                "right_keypoint": "right",
                "position_keypoint": "wild",
            },
            np.array(
                [
                    180.0,
                    180.0,
                    45.0,
                    -90.0,
                    np.rad2deg(np.pi / 2.0 + np.arcsin(1.0 / np.sqrt(5.0))),
                ]
            ),
            "compute_egocentric_angle",
            id="[E] Non-default position",
        ),
        pytest.param(
            "unit_square",
            "sample_position_array",
            {
                "left_keypoint": "left",
                "right_keypoint": "right",
            },
            np.array(
                [
                    0.0,
                    180.0,
                    float("nan"),
                    180.0,
                    float("nan"),
                ]
            ),
            "compute_egocentric_angle",
            id="[E] 0-approach vectors (nan returns)",
        ),
        pytest.param(
            "unit_square",
            "sample_position_array",
            {
                "left_keypoint": "left",
                "right_keypoint": "right",
                "boundary": True,
            },
            np.array(
                [
                    0.0,
                    180.0,
                    0.0,
                    180.0,
                    np.rad2deg(np.arccos(2.0 / np.sqrt(5.0))),
                ]
            ),
            "compute_egocentric_angle",
            id="[E] Force boundary calculations",
        ),
        pytest.param(
            "unit_square_with_hole",
            "sample_position_array",
            {
                "position_keypoint": "midpt",
            },
            np.array(
                [
                    -135.0,
                    0.0,
                    90.0,
                    -135.0,
                    180.0,
                ]
            ),
            "compute_allocentric_angle",
            id="[A] Default args",
        ),
        pytest.param(
            "unit_square",
            "sample_position_array",
            {
                "position_keypoint": "midpt",
            },
            np.array(
                [
                    -135.0,
                    0.0,
                    float("nan"),
                    -135.0,
                    float("nan"),
                ]
            ),
            "compute_allocentric_angle",
            id="[A] 0-approach vectors",
        ),
        pytest.param(
            "unit_square",
            "sample_position_array",
            {
                "position_keypoint": "midpt",
                "boundary": True,
            },
            np.array(
                [
                    -135.0,
                    0.0,
                    90.0,
                    -135.0,
                    180.0,
                ]
            ),
            "compute_allocentric_angle",
            id="[A] Force boundary calculation",
        ),
    ],
)
def test_ego_and_allocentric_angle_to_region(
    push_into_range,
    region: BaseRegionOfInterest,
    data: xr.DataArray,
    fn_args: dict[str, Any],
    expected_output: xr.DataArray | Exception,
    which_method: Literal[
        "compute_allocentric_angle", "compute_egocentric_angle"
    ],
    request,
) -> None:
    """Test computation of the egocentric and allocentric angle.

    Note, we only test functionality explicitly introduced in this method.
    Input arguments that are just handed to other functions are not explicitly
    tested here.

    Specifically;

    - ``camera_view``,
    - ``in_radians``,

    The ``angle_rotates`` argument is tested in all cases (signs should be
    reversed when toggling the argument).
    """
    if isinstance(region, str):
        region = request.getfixturevalue(region)
    if isinstance(data, str):
        data = request.getfixturevalue(data)
    if isinstance(expected_output, np.ndarray):
        expected_output = xr.DataArray(data=expected_output, dims=["time"])

    method = getattr(region, which_method)
    if which_method == "compute_egocentric_angle":
        other_vector_name = "forward"
    elif which_method == "compute_allocentric_angle":
        other_vector_name = "ref"

    if isinstance(expected_output, Exception):
        with pytest.raises(
            type(expected_output), match=re.escape(str(expected_output))
        ):
            method(data, **fn_args)
    else:
        angles = method(data, **fn_args)
        xr.testing.assert_allclose(angles, expected_output)

        # Check reversal of the angle convention
        if (
            fn_args.get("angle_rotates", f"approach to {other_vector_name}")
            == f"approach to {other_vector_name}"
        ):
            fn_args["angle_rotates"] = f"{other_vector_name} to approach"
        else:
            fn_args["angle_rotates"] = f"approach to {other_vector_name}"

        reverse_angles = push_into_range(method(data, **fn_args))
        xr.testing.assert_allclose(angles, push_into_range(-reverse_angles))


@pytest.fixture
def points_around_segment() -> xr.DataArray:
    """Points around the segment_of_y_equals_x.

    Data has (time, space, keypoints) dimensions, shape (, 2, 2).

    Keypoints are "left" and "right".

    time 1:
        left @ (0., 1.), right @ (0.05, 0.95).
        Fwd vector is (-1, -1).
    time 2:
        left @ (1., 0.), right @ (0.95, 0.05).
        Fwd vector is (-1, -1).
    time 3:
        left @ (1., 2.), right @ (1.05, 1.95).
        Fwd vector is (-1, -1).
        The egocentric angle will differ when using this point.
    """
    points = np.zeros(shape=(3, 2, 2))
    points[:, :, 0] = [
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 2.0],
    ]
    points[:, :, 1] = [
        [0.05, 0.95],
        [0.95, 0.05],
        [1.05, 1.95],
    ]
    return xr.DataArray(
        data=points,
        dims=["time", "space", "keypoints"],
        coords={
            "space": ["x", "y"],
            "keypoints": ["left", "right"],
        },
    )


def test_angle_to_support_plane(
    segment_of_y_equals_x: LineOfInterest,
    points_around_segment: xr.DataArray,
) -> None:
    expected_output = xr.DataArray(
        data=np.array([-90.0, -90.0, -90.0]), dims=["time"]
    )
    should_be_same_as_egocentric = expected_output.copy(
        data=[True, True, False], deep=True
    )

    angles_to_support = (
        segment_of_y_equals_x.compute_angle_to_support_plane_of_segment(
            points_around_segment, left_keypoint="left", right_keypoint="right"
        )
    )
    xr.testing.assert_allclose(expected_output, angles_to_support)

    egocentric_angles = segment_of_y_equals_x.compute_egocentric_angle(
        points_around_segment, left_keypoint="left", right_keypoint="right"
    )
    values_are_close = egocentric_angles.copy(
        data=np.isclose(egocentric_angles, angles_to_support), deep=True
    )
    xr.testing.assert_equal(should_be_same_as_egocentric, values_are_close)
