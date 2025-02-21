import re
from collections.abc import Iterable
from typing import Any

import numpy as np
import pytest
import xarray as xr

from movement.roi import LineOfInterest
from movement.roi.base import BaseRegionOfInterest


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
    ["region", "fn_args", "fn_kwargs", "expected_output", "egocentric"],
    [
        pytest.param(
            "unit_square_with_hole",
            [sample_position_array()],
            {
                "left_keypoint": "left",
                "right_keypoint": "right",
                "angle_rotates": "elephant to region",
            },
            ValueError("Unknown angle convention: elephant to region"),
            True,
            id="[E] Unknown angle convention",
        ),
        pytest.param(
            "unit_square_with_hole",
            [sample_position_array()],
            {
                "angle_rotates": "elephant to region",
            },
            ValueError("Unknown angle convention: elephant to region"),
            False,
            id="[A] Unknown angle convention",
        ),
        pytest.param(
            "unit_square_with_hole",
            [sample_position_array()],
            {
                "left_keypoint": "left",
                "right_keypoint": "right",
                "in_degrees": True,
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
            True,
            id="[E] Default args",
        ),
        pytest.param(
            "unit_square_with_hole",
            [sample_position_array()],
            {
                "left_keypoint": "left",
                "right_keypoint": "right",
                "position_keypoint": "wild",
                "in_degrees": True,
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
            True,
            id="[E] Non-default position",
        ),
        pytest.param(
            "unit_square",
            [sample_position_array()],
            {
                "left_keypoint": "left",
                "right_keypoint": "right",
                "in_degrees": True,
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
            True,
            id="[E] 0-approach vectors (nan returns)",
        ),
        pytest.param(
            "unit_square",
            [sample_position_array()],
            {
                "left_keypoint": "left",
                "right_keypoint": "right",
                "boundary_only": True,
                "in_degrees": True,
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
            True,
            id="[E] Force boundary calculations",
        ),
        pytest.param(
            "unit_square_with_hole",
            [
                sample_position_array()
                .sel(keypoints="midpt")
                .drop_vars("keypoints")
            ],
            {},
            np.deg2rad(
                np.array(
                    [
                        -135.0,
                        0.0,
                        90.0,
                        -135.0,
                        180.0,
                    ]
                )
            ),
            False,
            id="[A] Default args",
        ),
        pytest.param(
            "unit_square",
            [
                sample_position_array()
                .sel(keypoints="midpt")
                .drop_vars("keypoints")
            ],
            {},
            np.deg2rad(
                np.array(
                    [
                        -135.0,
                        0.0,
                        float("nan"),
                        -135.0,
                        float("nan"),
                    ]
                )
            ),
            False,
            id="[A] 0-approach vectors",
        ),
        pytest.param(
            "unit_square",
            [
                sample_position_array()
                .sel(keypoints="midpt")
                .drop_vars("keypoints")
            ],
            {"boundary_only": True},
            np.deg2rad(
                np.array(
                    [
                        -135.0,
                        0.0,
                        90.0,
                        -135.0,
                        180.0,
                    ]
                )
            ),
            False,
            id="[A] Force boundary calculation",
        ),
    ],
)
def test_ego_and_allocentric_angle_to_region(
    push_into_range,
    region: BaseRegionOfInterest,
    fn_args: Iterable[Any],
    fn_kwargs: dict[str, Any],
    expected_output: xr.DataArray | Exception,
    egocentric: bool,
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
    if isinstance(expected_output, np.ndarray):
        expected_output = xr.DataArray(data=expected_output, dims=["time"])

    if egocentric:
        which_method = "compute_egocentric_angle"
        other_vector_name = "forward"
    else:
        which_method = "compute_allocentric_angle_to_nearest_point"
        other_vector_name = "ref"
    method = getattr(region, which_method)

    if isinstance(expected_output, Exception):
        with pytest.raises(
            type(expected_output), match=re.escape(str(expected_output))
        ):
            method(*fn_args, **fn_kwargs)
    else:
        angles = method(*fn_args, **fn_kwargs)
        xr.testing.assert_allclose(angles, expected_output)

        # Check reversal of the angle convention
        if fn_kwargs.get("in_degrees", False):
            lower = -180.0
            upper = 180.0
        else:
            lower = -np.pi
            upper = np.pi
        if (
            fn_kwargs.get("angle_rotates", f"approach to {other_vector_name}")
            == f"approach to {other_vector_name}"
        ):
            fn_kwargs["angle_rotates"] = f"{other_vector_name} to approach"
        else:
            fn_kwargs["angle_rotates"] = f"approach to {other_vector_name}"

        reverse_angles = push_into_range(
            method(*fn_args, **fn_kwargs), lower=lower, upper=upper
        )
        xr.testing.assert_allclose(
            angles, push_into_range(-reverse_angles, lower=lower, upper=upper)
        )


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
