from collections.abc import Iterable
from typing import Any

import numpy as np
import pytest
import xarray as xr

from movement.kinematics import compute_forward_vector
from movement.roi import LineOfInterest
from movement.roi.base import BaseRegionOfInterest


def sample_position_array() -> xr.DataArray:
    """Return a simulated position array to test the egocentric angle.

    The data has time, space, and keypoints dimensions.

    The keypoints are left, right, midpt (midpoint), and wild.
    The midpt is the mean of the left and right keypoints.
    The wild keypoint is a point in the plane distinct from the midpt, and is
    used to test that the function respects the origin of the forward vector
    that the user provides (even if it is physically non-nonsensical).

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
        dims=["time", "space", "keypoint"],
        coords={
            "space": ["x", "y"],
            "keypoint": ["left", "right", "midpt", "wild"],
        },
    )


@pytest.mark.parametrize(
    ["region", "fn_args", "fn_kwargs", "expected_output", "egocentric"],
    [
        pytest.param(
            "unit_square_with_hole",
            [
                compute_forward_vector(
                    sample_position_array(), "left", "right"
                ),
                sample_position_array().sel(keypoint="midpt", drop=True),
            ],
            {"in_degrees": True},
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
            id="[Egocentric] Default args",
        ),
        pytest.param(
            "unit_square_with_hole",
            [
                compute_forward_vector(
                    sample_position_array(), "left", "right"
                ),
                sample_position_array().sel(keypoint="wild", drop=True),
            ],
            {"in_degrees": True},
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
            id="[Egocentric] Non-default position",
        ),
        pytest.param(
            "unit_square",
            [
                compute_forward_vector(
                    sample_position_array(), "left", "right"
                ),
                sample_position_array().sel(keypoint="midpt", drop=True),
            ],
            {"in_degrees": True},
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
            id="[Egocentric] 0-approach vectors (nan returns)",
        ),
        pytest.param(
            "unit_square",
            [
                compute_forward_vector(
                    sample_position_array(), "left", "right"
                ),
                sample_position_array().sel(keypoint="midpt", drop=True),
            ],
            {
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
            id="[Egocentric] Force boundary calculations",
        ),
        pytest.param(
            "unit_square_with_hole",
            [sample_position_array().sel(keypoint="midpt", drop=True)],
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
            id="[Allocentric] Default args",
        ),
        pytest.param(
            "unit_square",
            [sample_position_array().sel(keypoint="midpt", drop=True)],
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
            id="[Allocentric] 0-approach vectors",
        ),
        pytest.param(
            "unit_square",
            [sample_position_array().sel(keypoint="midpt", drop=True)],
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
            id="[Allocentric] Force boundary calculation",
        ),
    ],
)
def test_ego_and_allocentric_angle_to_region(
    region: BaseRegionOfInterest,
    fn_args: Iterable[Any],
    fn_kwargs: dict[str, Any],
    expected_output: xr.DataArray,
    egocentric: bool,
    request,
) -> None:
    """Test computation of the egocentric and allocentric angle.

    Note, we only test functionality explicitly introduced in this method.
    Input arguments that are just handed to other functions (i.e.,
    ``in_degrees``), are not explicitly tested here.
    """
    if isinstance(region, str):
        region = request.getfixturevalue(region)
    if isinstance(expected_output, np.ndarray):
        expected_output = xr.DataArray(data=expected_output, dims=["time"])

    if egocentric:
        which_method = "compute_egocentric_angle_to_nearest_point"
    else:
        which_method = "compute_allocentric_angle_to_nearest_point"
    method = getattr(region, which_method)

    angles = method(*fn_args, **fn_kwargs)
    xr.testing.assert_allclose(angles, expected_output)


@pytest.fixture
def points_around_segment() -> xr.DataArray:
    """Sample points on either side of the segment y=x.

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
        dims=["time", "space", "keypoint"],
        coords={
            "space": ["x", "y"],
            "keypoint": ["left", "right"],
        },
    )


def test_angle_to_normal(
    segment_of_y_equals_x: LineOfInterest,
    points_around_segment: xr.DataArray,
) -> None:
    """Test the angle with normal vector computation.

    This method checks two things:

    - The compute_angle_to_normal method returns the correct angle, and
    - The method agrees with the egocentric angle computation, in the cases
    that the two calculations should return the same value (IE when the
    approach vector is the normal to the segment). And that the
    returned angles are different otherwise.
    """
    expected_output = xr.DataArray(
        data=np.deg2rad([-90.0, -90.0, -90.0]), dims=["time"]
    )
    should_be_same_as_egocentric = expected_output.copy(
        data=[True, True, False], deep=True
    )

    fwd_vector = compute_forward_vector(points_around_segment, "left", "right")
    positions = points_around_segment.mean(dim="keypoint")
    angles_to_support = segment_of_y_equals_x.compute_angle_to_normal(
        fwd_vector, positions
    )
    xr.testing.assert_allclose(expected_output, angles_to_support)

    egocentric_angles = (
        segment_of_y_equals_x.compute_egocentric_angle_to_nearest_point(
            fwd_vector, positions
        )
    )
    values_are_close = egocentric_angles.copy(
        data=np.isclose(egocentric_angles, angles_to_support), deep=True
    )
    xr.testing.assert_equal(should_be_same_as_egocentric, values_are_close)
