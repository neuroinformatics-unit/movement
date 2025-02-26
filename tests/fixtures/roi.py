import numpy as np
import pytest
import xarray as xr

from movement.roi import LineOfInterest
from movement.roi.polygon import PolygonOfInterest


@pytest.fixture
def segment_of_y_equals_x() -> LineOfInterest:
    """Line segment from (0,0) to (1,1)."""
    return LineOfInterest([(0, 0), (1, 1)])


@pytest.fixture()
def unit_square_pts() -> np.ndarray:
    """Points that define the 4 corners of a unit-length square.

    The points have the lower-left corner positioned at (0,0).
    """
    return np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=float,
    )


@pytest.fixture()
def unit_square_hole(unit_square_pts: np.ndarray) -> np.ndarray:
    """Hole in the shape of a 0.5 side-length square centred on (0.5, 0.5)."""
    return 0.25 + (unit_square_pts.copy() * 0.5)


@pytest.fixture
def unit_square(unit_square_pts: xr.DataArray) -> PolygonOfInterest:
    """Square of unit side-length centred on (0.5, 0.5)."""
    return PolygonOfInterest(unit_square_pts, name="Unit square")


@pytest.fixture
def unit_square_with_hole(
    unit_square_pts: xr.DataArray, unit_square_hole: xr.DataArray
) -> PolygonOfInterest:
    """Square of unit side length with an internal hole.

    The "outer" square is centred on (0.5, 0.5) and has side length 1.
    The "inner" square, or hole, is centred on (0.5, 0.5) and has side length
    0.5.
    """
    return PolygonOfInterest(
        unit_square_pts, holes=[unit_square_hole], name="Unit square with hole"
    )
