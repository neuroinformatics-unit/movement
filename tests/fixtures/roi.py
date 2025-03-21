import numpy as np
import pytest
import xarray as xr

from movement.roi import LineOfInterest, PolygonOfInterest


@pytest.fixture
def segment_of_y_equals_x() -> LineOfInterest:
    """Line segment from (0,0) to (1,1)."""
    return LineOfInterest([(0, 0), (1, 1)])


@pytest.fixture()
def unit_square_pts() -> np.ndarray:
    """Points that define the 4 corners of a unit-length square.

    The points have the lower-left corner positioned at (0,0),
    and run clockwise around the centre of the would-be square.
    """
    return np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, 0.0],
        ],
        dtype=float,
    )


@pytest.fixture()
def triangle_pts():
    """Vertices of a right-angled triangle."""
    return [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]


@pytest.fixture()
def unit_square_hole(unit_square_pts: np.ndarray) -> np.ndarray:
    """Hole in the shape of a 0.5 side-length square centred on (0.5, 0.5)."""
    return 0.25 + (unit_square_pts.copy() * 0.5)


@pytest.fixture
def unit_square(unit_square_pts: xr.DataArray) -> PolygonOfInterest:
    return PolygonOfInterest(unit_square_pts, name="Unit square")


@pytest.fixture
def unit_square_with_hole(
    unit_square_pts: xr.DataArray, unit_square_hole: xr.DataArray
) -> PolygonOfInterest:
    return PolygonOfInterest(
        unit_square_pts, holes=[unit_square_hole], name="Unit square with hole"
    )


@pytest.fixture()
def triangle(triangle_pts) -> PolygonOfInterest:
    """Triangle."""
    return PolygonOfInterest(triangle_pts, name="triangle")


@pytest.fixture()
def triangle_different_name(triangle_pts) -> PolygonOfInterest:
    """Triangle with a different name."""
    return PolygonOfInterest(triangle_pts, name="pizza_slice")


@pytest.fixture()
def triangle_moved_01(triangle_pts) -> PolygonOfInterest:
    """Triangle moved by 0.01 on the x and y axis."""
    return PolygonOfInterest(
        [(x + 0.01, y + 0.01) for x, y in triangle_pts], name="triangle"
    )


@pytest.fixture()
def triangle_moved_100(triangle_pts) -> PolygonOfInterest:
    """Triangle moved by 1.00 on the x and y axis."""
    return PolygonOfInterest(
        [(x + 1.0, y + 1.0) for x, y in triangle_pts], name="triangle"
    )
