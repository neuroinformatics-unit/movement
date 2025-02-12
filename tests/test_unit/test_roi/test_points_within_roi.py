import numpy as np
import pytest
import xarray as xr

from movement.roi.polygon import PolygonOfInterest


@pytest.fixture
def holey_polygon(request) -> PolygonOfInterest:
    """Fixture for a polygon with a hole.

    RoI is a square with a 0.5 by 0.5 hole in the middle.
    (0.0, 0.0) -> (1.0, 0.0) -> (1.0, 1.0) -> (0.0, 1.0) -> (0.0, 0.0)
    """
    exterior_boundary = request.getfixturevalue("unit_square_pts")
    interior_boundary = 0.25 + (exterior_boundary.copy() * 0.5)
    return PolygonOfInterest(exterior_boundary, holes=[interior_boundary])


@pytest.mark.parametrize(
    ["point", "inside"],
    [
        pytest.param([0, 0], True, id="point on boundary"),
        pytest.param([0.5, 0.5], False, id="point inside hole"),
        pytest.param([0.1, 0.1], True, id="point inside roi"),
        pytest.param([2.0, 2.0], False, id="point outside roi"),
        pytest.param([0.5, 0.5, 0.5], False, id="3D point"),
    ],
)
def test_point_within_roi(holey_polygon, point, inside) -> None:
    """Test whether a point is within RoI."""
    assert holey_polygon.point_is_inside(point) == inside


@pytest.mark.parametrize(
    ["points", "expected"],
    [
        pytest.param(
            xr.DataArray(
                np.array([[0.15, 0.15], [0.1, 0.1], [0.80, 0.80]]),
                dims=["points", "space"],
            ),
            xr.DataArray([True, True, True], dims=["points"]),
            id="3 points inside",
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
                np.array([[2, 2], [-2, -2]]),
                dims=["points", "space"],
            ),
            xr.DataArray([False, False], dims=["points"]),
            id="2 points outside hole",
        ),
    ],
)
def test_points_within_roi(holey_polygon, points, expected) -> None:
    """Test whether points are within RoI."""
    xr.testing.assert_equal(
        holey_polygon.point_is_inside(points),
        expected,
    )


def create_data_array(shape, dims):
    """Create an xarray DataArray with zeros."""
    return xr.DataArray(np.zeros(shape), dims=dims)


@pytest.mark.parametrize(
    ["points", "expected_shape", "expected_dims"],
    [
        pytest.param(
            create_data_array((4, 2), ["points", "space"]),
            (4,),
            ("points",),
            id="points (4)",
        ),
        pytest.param(
            create_data_array((2, 2, 2), ["time", "points", "space"]),
            (2, 2),
            ("time", "points"),
            id="time (2), points (2)",
        ),
        pytest.param(
            create_data_array(
                (2, 2, 2, 5), ["time", "points", "space", "individuals"]
            ),
            (2, 2, 5),
            ("time", "points", "individuals"),
            id="time (2), points (2), individuals (5)",
        ),
    ],
)
def test_shape_dims(
    holey_polygon, points, expected_shape, expected_dims
) -> None:
    """Check the shape and dims of the result.

    The space dimension should have collapsed.
    """
    result = holey_polygon.point_is_inside(points)
    assert result.shape == expected_shape
    assert result.dims == expected_dims
