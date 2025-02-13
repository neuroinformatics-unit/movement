import re
from typing import Any

import numpy as np
import pytest
import xarray as xr

from movement.roi.base import BaseRegionOfInterest


@pytest.fixture()
def points_in_the_plane() -> xr.DataArray:
    """Define a collection of points, used to test the egocentric angle.

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
    ["region", "data", "fn_args", "expected_output"],
    [
        pytest.param(
            "unit_square_with_hole",
            "points_in_the_plane",
            {
                "left_keypoint": "left",
                "right_keypoint": "right",
                "angle_rotates": "elephant to region",
            },
            ValueError("Unknown angle convention: elephant to region"),
            id="Unknown angle convention (checked before other failures)",
        ),
        pytest.param(
            "unit_square_with_hole",
            "points_in_the_plane",
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
            id="Default args",
        ),
        pytest.param(
            "unit_square_with_hole",
            "points_in_the_plane",
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
            id="Non-default position",
        ),
        pytest.param(
            "unit_square",
            "points_in_the_plane",
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
            id="0-approach vectors (nan returns)",
        ),
        pytest.param(
            "unit_square",
            "points_in_the_plane",
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
            id="Force boundary calculations",
        ),
    ],
)
def test_egocentric_angle(
    push_into_range,
    region: BaseRegionOfInterest,
    data: xr.DataArray,
    fn_args: dict[str, Any],
    expected_output: xr.DataArray | Exception,
    request,
) -> None:
    """Test computation of the egocentric angle.

    Note, we only test functionality explicitly introduced in this method.
    Input arguments that are just handed to other functions are not explicitly
    tested here.

    Specifically;

    - ``approach_direction``,
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

    if isinstance(expected_output, Exception):
        with pytest.raises(
            type(expected_output), match=re.escape(str(expected_output))
        ):
            region.compute_egocentric_angle(data, **fn_args)
    else:
        angles = region.compute_egocentric_angle(data, **fn_args)

        xr.testing.assert_allclose(angles, expected_output)

        # Check reversal of the angle convention
        if (
            fn_args.get("angle_rotates", "approach to forward")
            == "approach to forward"
        ):
            fn_args["angle_rotates"] = "forward to approach"
        else:
            fn_args["angle_rotates"] = "approach to forward"
        reverse_angles = push_into_range(
            region.compute_egocentric_angle(data, **fn_args)
        )

        xr.testing.assert_allclose(angles, push_into_range(-reverse_angles))
