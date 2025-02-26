import re
from typing import Any

import numpy as np
import pytest
import xarray as xr

from movement.roi.base import BaseRegionOfInterest
from movement.roi.line import LineOfInterest


@pytest.fixture
def sample_target_points() -> dict[str, np.ndarray]:
    """Sample 2D trajectory data."""
    return xr.DataArray(
        np.array(
            [
                [-0.5, 0.50],
                [0.00, 0.50],
                [0.40, 0.45],
                [2.00, 1.00],
                [0.40, 0.75],
                [0.95, 0.90],
                [0.80, 0.76],
            ]
        ),
        dims=["time", "space"],
        coords={"space": ["x", "y"]},
    )


@pytest.fixture()
def unit_line_in_x() -> LineOfInterest:
    return LineOfInterest([[0.0, 0.0], [1.0, 0.0]])


@pytest.mark.parametrize(
    ["region", "fn_kwargs", "expected_distances"],
    [
        pytest.param(
            "unit_square_with_hole",
            {"boundary_only": True},
            np.array(
                [
                    0.5,
                    0.0,
                    0.15,
                    1.0,
                    0.0,
                    0.05,
                    np.sqrt((1.0 / 20.0) ** 2 + (1.0 / 100.0) ** 2),
                ]
            ),
            id="Unit square w/ hole, boundary",
        ),
        pytest.param(
            "unit_square_with_hole",
            {},
            np.array([0.5, 0.0, 0.15, 1.0, 0.0, 0.0, 0.0]),
            id="Unit square w/ hole, whole region",
        ),
        pytest.param(
            "unit_line_in_x",
            {},
            np.array(
                [0.5 * np.sqrt(2.0), 0.5, 0.45, np.sqrt(2.0), 0.75, 0.9, 0.76]
            ),
            id="Unit line in x",
        ),
        pytest.param(
            "unit_line_in_x",
            {"boundary_only": True},
            np.array(
                [
                    0.5 * np.sqrt(2.0),
                    0.5,
                    np.sqrt(0.4**2 + 0.45**2),
                    np.sqrt(2.0),
                    np.sqrt(0.4**2 + 0.75**2),
                    np.sqrt(0.05**2 + 0.9**2),
                    np.sqrt(0.2**2 + 0.76**2),
                ]
            ),
            id="Unit line in x, endpoints only",
        ),
    ],
)
def test_distance_point_to_region(
    region: BaseRegionOfInterest,
    sample_target_points: xr.DataArray,
    fn_kwargs: dict[str, Any],
    expected_distances: xr.DataArray,
    request,
) -> None:
    if isinstance(region, str):
        region = request.getfixturevalue(region)
    if isinstance(expected_distances, np.ndarray):
        expected_distances = xr.DataArray(
            data=expected_distances, dims=["time"]
        )

    computed_distances = region.compute_distance_to(
        sample_target_points, **fn_kwargs
    )

    xr.testing.assert_allclose(computed_distances, expected_distances)


@pytest.mark.parametrize(
    ["region", "other_fn_args", "expected_output"],
    [
        pytest.param(
            "unit_square",
            {"boundary_only": True},
            np.array(
                [
                    [0.00, 0.50],
                    [0.00, 0.50],
                    [0.00, 0.45],
                    [1.00, 1.00],
                    [0.40, 1.00],
                    [1.00, 0.90],
                    [1.00, 0.76],
                ]
            ),
            id="Unit square, boundary only",
        ),
        pytest.param(
            "unit_square",
            {},
            np.array(
                [
                    [0.00, 0.50],
                    [0.00, 0.50],
                    [0.40, 0.45],
                    [1.00, 1.00],
                    [0.40, 0.75],
                    [0.95, 0.90],
                    [0.80, 0.76],
                ]
            ),
            id="Unit square, whole region",
        ),
        pytest.param(
            "unit_square_with_hole",
            {"boundary_only": True},
            np.array(
                [
                    [0.00, 0.50],
                    [0.00, 0.50],
                    [0.25, 0.45],
                    [1.00, 1.00],
                    [0.40, 0.75],
                    [1.00, 0.90],
                    [0.75, 0.75],
                ]
            ),
            id="Unit square w/ hole, boundary only",
        ),
        pytest.param(
            "unit_square_with_hole",
            {},
            np.array(
                [
                    [0.00, 0.50],
                    [0.00, 0.50],
                    [0.25, 0.45],
                    [1.00, 1.00],
                    [0.40, 0.75],
                    [0.95, 0.90],
                    [0.80, 0.76],
                ]
            ),
            id="Unit square w/ hole, whole region",
        ),
        pytest.param(
            "unit_line_in_x",
            {},
            np.array(
                [
                    [0.00, 0.00],
                    [0.00, 0.00],
                    [0.40, 0.00],
                    [1.00, 0.00],
                    [0.40, 0.00],
                    [0.95, 0.00],
                    [0.80, 0.00],
                ]
            ),
            id="Line, whole region",
        ),
        pytest.param(
            "unit_line_in_x",
            {"boundary_only": True},
            np.array(
                [
                    [0.00, 0.00],
                    [0.00, 0.00],
                    [0.00, 0.00],
                    [1.00, 0.00],
                    [0.00, 0.00],
                    [1.00, 0.00],
                    [1.00, 0.00],
                ]
            ),
            id="Line, boundary only",
        ),
    ],
)
def test_nearest_point_to(
    region: BaseRegionOfInterest,
    sample_target_points: xr.DataArray,
    other_fn_args: dict[str, Any],
    expected_output: xr.DataArray,
    request,
) -> None:
    if isinstance(region, str):
        region = request.getfixturevalue(region)
    if isinstance(expected_output, str):
        expected_output = request.get(expected_output)
    elif isinstance(expected_output, np.ndarray):
        expected_output = xr.DataArray(
            expected_output,
            dims=["time", "nearest point"],
        )

    nearest_points = region.compute_nearest_point_to(
        sample_target_points, **other_fn_args
    )

    xr.testing.assert_allclose(nearest_points, expected_output)


@pytest.mark.parametrize(
    ["region", "position", "fn_kwargs", "possible_nearest_points"],
    [
        pytest.param(
            "unit_square",
            [0.5, 0.5],
            {"boundary_only": True},
            [
                np.array([0.0, 0.5]),
                np.array([0.5, 0.0]),
                np.array([1.0, 0.5]),
                np.array([0.5, 1.0]),
            ],
            id="Centre of the unit square",
        ),
        pytest.param(
            "unit_line_in_x",
            [0.5, 0.0],
            {"boundary_only": True},
            [
                np.array([0.0, 0.0]),
                np.array([1.0, 0.0]),
            ],
            id="Boundary of a line",
        ),
    ],
)
def test_nearest_point_to_tie_breaks(
    region: BaseRegionOfInterest,
    position: np.ndarray,
    fn_kwargs: dict[str, Any],
    possible_nearest_points: list[np.ndarray],
    request,
) -> None:
    """Check behaviour when points are tied for nearest.

    This can only occur when we have a Polygonal region, or a multi-line 1D
    region. In this case, there may be multiple points in the region of
    interest that are tied for closest. ``shapely`` does not actually document
    how it breaks ties here, but we can at least check that it identifies one
    of the possible correct points.
    """
    if isinstance(region, str):
        region = request.getfixturevalue(region)
    if not isinstance(position, np.ndarray | xr.DataArray):
        position = np.array(position)

    nearest_point_found = region.compute_nearest_point_to(
        position, **fn_kwargs
    )

    sq_dist_to_nearest_pt = np.sum((nearest_point_found - position) ** 2)

    n_matches = 0
    for possibility in possible_nearest_points:
        # All possibilities should be approximately the same distance away
        # from the position
        assert np.isclose(
            np.sum((possibility - position) ** 2), sq_dist_to_nearest_pt
        )
        # We should match at least one possibility,
        # track to see if we do.
        if np.isclose(nearest_point_found, possibility).all():
            n_matches += 1
    assert n_matches == 1


@pytest.mark.parametrize(
    ["region", "point", "other_fn_args", "expected_output"],
    [
        pytest.param(
            "unit_square",
            (-0.5, 0.0),
            {"unit": True},
            np.array([1.0, 0.0]),
            id="(-0.5, 0.0) -> unit square",
        ),
        pytest.param(
            LineOfInterest([(0.0, 0.0), (1.0, 0.0)]),
            (0.1, 0.5),
            {"unit": True},
            np.array([0.0, -1.0]),
            id="(0.1, 0.5) -> +ve x ray",
        ),
        pytest.param(
            "unit_square",
            (-0.5, 0.0),
            {"unit": False},
            np.array([0.5, 0.0]),
            id="Don't normalise output",
        ),
        pytest.param(
            "unit_square",
            (0.5, 0.5),
            {"unit": True},
            np.array([0.0, 0.0]),
            id="Interior point returns 0 vector",
        ),
        pytest.param(
            "unit_square",
            (0.25, 0.35),
            {"boundary_only": True, "unit": True},
            np.array([-1.0, 0.0]),
            id="Boundary, polygon",
        ),
        pytest.param(
            LineOfInterest([(0.0, 0.0), (1.0, 0.0)]),
            (0.1, 0.5),
            {"boundary_only": True, "unit": True},
            np.array([-0.1, -0.5]) / np.sqrt(0.1**2 + 0.5**2),
            id="Boundary, line",
        ),
    ],
)
def test_approach_vector(
    region: BaseRegionOfInterest,
    point: xr.DataArray,
    other_fn_args: dict[str, Any],
    expected_output: np.ndarray | Exception,
    request,
) -> None:
    if isinstance(region, str):
        region = request.getfixturevalue(region)

    if isinstance(expected_output, Exception):
        with pytest.raises(
            type(expected_output), match=re.escape(str(expected_output))
        ):
            region.compute_approach_vector(point, **other_fn_args)
    else:
        vector_to = region.compute_approach_vector(point, **other_fn_args)
        assert np.allclose(vector_to, expected_output)
