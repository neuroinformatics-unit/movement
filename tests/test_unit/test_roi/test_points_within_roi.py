import numpy as np
import pytest
import xarray as xr

from movement.roi.line import LineOfInterest


@pytest.fixture
def diagonal_line() -> LineOfInterest:
    """Fixture for a line.

    RoI is a diagonal line from the origin to (1, 1).
    """
    return LineOfInterest([(0, 0), (1, 1)])


@pytest.mark.parametrize(
    ["point", "include_boundary", "inside"],
    [
        pytest.param([0, 0], True, True, id="on starting point"),
        pytest.param([0, 0], False, False, id="on excluded starting point"),
        pytest.param([0.5, 0.5], True, True, id="inside LoI"),
        pytest.param(
            [0.5, 0.5], False, True, id="inside LoI (exclude boundary)"
        ),
        pytest.param([2.0, 2.0], True, False, id="outside LoI"),
        pytest.param([0.5, 0.5, 0.5], True, True, id="3D point inside LoI"),
        pytest.param([0.1, 0.2, 0.5], True, False, id="3D point outside LoI"),
    ],
)
def test_point_within_line(
    diagonal_line, point, include_boundary, inside
) -> None:
    """Test whether a point is on a line.

    The boundaries of a line are the end points.
    """
    assert diagonal_line.contains_point(point, include_boundary) == inside


@pytest.mark.parametrize(
    ["point", "include_boundary", "inside"],
    [
        pytest.param([0, 0], True, True, id="on exterior boundary"),
        pytest.param([0, 0], False, False, id="on excluded exterior boundary"),
        pytest.param([0.25, 0.25], True, True, id="on hole boundary"),
        pytest.param(
            [0.25, 0.25], False, False, id="on excluded hole boundary"
        ),
        pytest.param([0.5, 0.5], True, False, id="inside hole"),
        pytest.param([0.1, 0.1], True, True, id="inside RoI"),
        pytest.param([2.0, 2.0], True, False, id="outside RoI"),
        pytest.param([0.5, 0.5, 0.5], True, False, id="3D point inside hole"),
        pytest.param([0.1, 0.1, 0.5], True, True, id="3D point inside RoI"),
    ],
)
def test_point_within_polygon(
    unit_square_with_hole, point, include_boundary, inside
) -> None:
    """Test whether a point is within RoI."""
    assert (
        unit_square_with_hole.contains_point(point, include_boundary) == inside
    )


@pytest.mark.parametrize(
    ["points", "expected"],
    [
        pytest.param(
            xr.DataArray(
                np.array([[0.15, 0.15], [0.1, 0.1], [0.80, 0.80]]),
                dims=["points", "space"],
            ),
            xr.DataArray([True, True, True], dims=["points"]),
            id="3 points inside RoI",
        ),
        pytest.param(
            xr.DataArray(
                np.array([[0.55, 0.55], [0.5, 0.5], [0.7, 0.7], [0.6, 0.6]]),
                dims=["points", "space"],
            ),
            xr.DataArray([False, False, False, False], dims=["points"]),
            id="4 points inside hole",
        ),
        pytest.param(
            xr.DataArray(
                np.array([[0.55, 0.55], [0.1, 0.1], [0.7, 0.7], [0, 0]]),
                dims=["points", "space"],
            ),
            xr.DataArray([False, True, False, True], dims=["points"]),
            id="2 points inside hole, 1 point inside RoI, 1 on RoI boundary",
        ),
        pytest.param(
            xr.DataArray(
                np.array([[2, 2], [-2, -2]]),
                dims=["points", "space"],
            ),
            xr.DataArray([False, False], dims=["points"]),
            id="2 points outside RoI",
        ),
    ],
)
def test_points_within_polygon(
    unit_square_with_hole, points, expected
) -> None:
    """Test whether points (supplied as xr.DataArray) are within a polygon."""
    xr.testing.assert_equal(
        unit_square_with_hole.contains_point(points),
        expected,
    )


@pytest.mark.parametrize(
    ["points", "expected_shape", "expected_dims"],
    [
        pytest.param(
            xr.DataArray(np.zeros((4, 2)), dims=["points", "space"]),
            (4,),
            ("points",),
            id="points (4)",
        ),
        pytest.param(
            xr.DataArray(
                np.zeros((2, 2, 2)), dims=["time", "points", "space"]
            ),
            (2, 2),
            ("time", "points"),
            id="time (2), points (2)",
        ),
        pytest.param(
            xr.DataArray(
                np.zeros((2, 2, 2, 5)),
                dims=["time", "points", "space", "individual"],
            ),
            (2, 2, 5),
            ("time", "points", "individual"),
            id="time (2), points (2), individual (5)",
        ),
    ],
)
def test_shape_dims(
    unit_square_with_hole, points, expected_shape, expected_dims
) -> None:
    """Check the shape and dims of the result.

    The space dimension should have collapsed.
    """
    result = unit_square_with_hole.contains_point(points)
    assert result.shape == expected_shape
    assert result.dims == expected_dims
