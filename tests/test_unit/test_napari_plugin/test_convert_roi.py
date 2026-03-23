"""Unit tests for napari shape → movement RoI conversion."""

import numpy as np
import pytest
import shapely

from movement.napari.convert_roi import napari_shape_to_roi
from movement.roi import LineOfInterest, PolygonOfInterest

# ---------------------------------------------------------------------------
# Fixtures — napari (y, x) shape data
#
# Where possible, the spatial values are derived from the shared RoI fixtures
# (unit_square_pts) by swapping columns, so that converting back should
# reproduce those same point-sets.
# ---------------------------------------------------------------------------


@pytest.fixture
def line_yx():
    """Two-point line in napari (y, x) convention."""
    return np.array([[2.0, 1.0], [4.0, 3.0]])


@pytest.fixture
def path_yx():
    """Three-point path in napari (y, x) convention."""
    return np.array([[0.0, 0.0], [2.0, 1.0], [1.0, 3.0]])


@pytest.fixture
def square_yx(unit_square_pts):
    """Unit-square corners in napari (y, x) convention.

    Derived by swapping columns of the shared ``unit_square_pts`` fixture,
    which is in movement (x, y) convention.
    """
    return unit_square_pts[:, ::-1]


@pytest.fixture
def nonsquare_rect_yx():
    """Non-square rectangle in napari (y, x)."""
    return np.array([[0.0, 0.0], [0.0, 3.0], [1.0, 3.0], [1.0, 0.0]])


@pytest.fixture
def ellipse_yx():
    """Axis-aligned ellipse in napari (y, x) convention.

    Centre (cy=5, cx=5), semi-axes ry=3, rx=2.
    The 4 cardinal points are: top, right, bottom, left.
    """
    return np.array([[2.0, 5.0], [5.0, 7.0], [8.0, 5.0], [5.0, 3.0]])


# ---------------------------------------------------------------------------
# RoI type dispatch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ["shape_type", "data_fixture", "expected_type"],
    [
        pytest.param(
            "line", "line_yx", LineOfInterest, id="line → LineOfInterest"
        ),
        pytest.param(
            "path", "path_yx", LineOfInterest, id="path → LineOfInterest"
        ),
        pytest.param(
            "polygon",
            "square_yx",
            PolygonOfInterest,
            id="polygon → PolygonOfInterest",
        ),
        pytest.param(
            "rectangle",
            "square_yx",
            PolygonOfInterest,
            id="rectangle → PolygonOfInterest",
        ),
        pytest.param(
            "ellipse",
            "ellipse_yx",
            PolygonOfInterest,
            id="ellipse → PolygonOfInterest",
        ),
    ],
)
def test_roi_type(shape_type, data_fixture, expected_type, request):
    """Each napari shape type maps to the correct movement RoI class."""
    data = request.getfixturevalue(data_fixture)
    assert isinstance(napari_shape_to_roi(data, shape_type), expected_type)


# ---------------------------------------------------------------------------
# Name assignment
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ["name", "expected_name"],
    [
        pytest.param("my boundary", "my boundary", id="explicit name"),
        pytest.param(None, "Un-named region", id="None uses default name"),
    ],
)
def test_name_assignment(line_yx, name, expected_name):
    roi = napari_shape_to_roi(line_yx, "line", name=name)
    assert roi.name == expected_name


# ---------------------------------------------------------------------------
# Coordinate swap: napari (y, x) → movement (x, y)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ["shape_type", "data_fixture", "expected_geometry"],
    [
        pytest.param(
            "line",
            "line_yx",
            shapely.LineString([[1.0, 2.0], [3.0, 4.0]]),
            id="line: (y,x) coords are swapped to (x,y)",
        ),
        pytest.param(
            "polygon",
            "square_yx",
            shapely.Polygon([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]]),
            id="polygon: unit-square round-trip after swap",
        ),
    ],
)
def test_coordinate_swap(shape_type, data_fixture, expected_geometry, request):
    """Napari (y, x) coordinates are converted to movement (x, y)."""
    data = request.getfixturevalue(data_fixture)
    roi = napari_shape_to_roi(data, shape_type)
    assert shapely.normalize(roi.region) == shapely.normalize(
        expected_geometry
    )


# ---------------------------------------------------------------------------
# Region bounds (also covers ellipse approximation accuracy)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ["shape_type", "data_fixture", "expected_bounds"],
    [
        pytest.param(
            "rectangle",
            "nonsquare_rect_yx",
            (0.0, 0.0, 3.0, 1.0),
            id="rectangle: napari x∈[0,3], y∈[0,1] → x∈[0,3], y∈[0,1]",
        ),
        pytest.param(
            "ellipse",
            "ellipse_yx",
            # centre (5, 5), semi_x=2, semi_y=3 → (5-2, 5-3, 5+2, 5+3)
            (3.0, 2.0, 7.0, 8.0),
            id="ellipse: bounds match semi-axes (abs tolerance 0.01)",
        ),
    ],
)
def test_region_bounds(shape_type, data_fixture, expected_bounds, request):
    """Region bounding box matches expected (minx, miny, maxx, maxy)."""
    data = request.getfixturevalue(data_fixture)
    roi = napari_shape_to_roi(data, shape_type)
    assert roi.region.bounds == pytest.approx(expected_bounds, abs=0.01)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ["data", "shape_type", "match"],
    [
        pytest.param(
            np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]),
            "line",
            "3 columns",
            id="3-column data raises ValueError",
        ),
        pytest.param(
            np.array([[0.0, 1.0], [2.0, 3.0]]),
            "circle",  # type: ignore[arg-type]
            "Unrecognized napari shape type",
            id="unknown shape_type raises ValueError",
        ),
    ],
)
def test_raises_on_invalid_input(data, shape_type, match):
    """Invalid inputs raise ValueError with an informative message."""
    with pytest.raises(ValueError, match=match):
        napari_shape_to_roi(data, shape_type)
