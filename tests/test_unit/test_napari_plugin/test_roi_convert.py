"""Unit tests for napari shape → movement ROI conversion."""

import re

import numpy as np
import pytest

from movement.napari.roi_convert import napari_shape_to_roi
from movement.roi import LineOfInterest, PolygonOfInterest

# ---------------------------------------------------------------------------
# Fixtures — napari (y, x) shape data
#
# Where possible, the spatial values are derived from the shared ROI fixtures
# (unit_square_pts, triangle_pts) by swapping columns, so that converting
# back should reproduce those same point-sets.
# ---------------------------------------------------------------------------


@pytest.fixture
def line_yx():
    """Two-point line in napari (y, x) convention.

    In movement (x, y) this represents (1, 2) → (3, 4).
    """
    return np.array([[2.0, 1.0], [4.0, 3.0]])


@pytest.fixture
def line_xy_expected():
    """Return expected (x, y) coordinates after converting ``line_yx``."""
    return np.array([[1.0, 2.0], [3.0, 4.0]])


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
    """Non-square rectangle in napari (y, x) so the coordinate swap is visible.

    napari y ∈ [0, 1], napari x ∈ [0, 3].
    After swap: movement x ∈ [0, 3], movement y ∈ [0, 1].
    """
    return np.array([[0.0, 0.0], [0.0, 3.0], [1.0, 3.0], [1.0, 0.0]])


@pytest.fixture
def ellipse_yx():
    """Axis-aligned ellipse in napari (y, x) convention.

    Centre (cy=5, cx=5), semi-axes ry=3, rx=2.
    The 4 cardinal points are: top, right, bottom, left.
    """
    return np.array([[2.0, 5.0], [5.0, 7.0], [8.0, 5.0], [5.0, 3.0]])


@pytest.fixture
def ellipse_centre_xy():
    """Centre of ``ellipse_yx`` in movement (x, y) convention."""
    return np.array([5.0, 5.0])


@pytest.fixture
def ellipse_semi_axes():
    """(semi_x, semi_y) of ``ellipse_yx``."""
    return 2.0, 3.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sorted_coords(coords) -> np.ndarray:
    """Sort coordinate array row-wise for order-agnostic comparison."""
    arr = np.array(list(coords))
    return arr[np.lexsort(arr.T[::-1])]


# ---------------------------------------------------------------------------
# napari_shape_to_roi — ROI type
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape_type, data_fixture, expected_type",
    [
        ("line", "line_yx", LineOfInterest),
        ("path", "path_yx", LineOfInterest),
        ("polygon", "square_yx", PolygonOfInterest),
        ("rectangle", "square_yx", PolygonOfInterest),
    ],
)
def test_returns_correct_roi_type(
    shape_type, data_fixture, expected_type, request
):
    data = request.getfixturevalue(data_fixture)
    roi = napari_shape_to_roi(data, shape_type)
    assert isinstance(roi, expected_type)


def test_name_is_passed_through(line_yx):
    roi = napari_shape_to_roi(line_yx, "line", name="my boundary")
    assert roi.name == "my boundary"


def test_default_name_when_none(line_yx):
    roi = napari_shape_to_roi(line_yx, "line", name=None)
    assert roi.name == "Un-named region"


# ---------------------------------------------------------------------------
# Coordinate swap: napari (y, x) → movement (x, y)
# ---------------------------------------------------------------------------


def test_line_coordinates_are_swapped(line_yx, line_xy_expected):
    roi = napari_shape_to_roi(line_yx, "line")
    np.testing.assert_array_almost_equal(
        _sorted_coords(roi.coords),
        _sorted_coords(line_xy_expected),
    )


def test_polygon_bounding_box_reflects_coordinate_swap(nonsquare_rect_yx):
    """Bounding box should reflect the (y, x) → (x, y) swap."""
    roi = napari_shape_to_roi(nonsquare_rect_yx, "rectangle")
    minx, miny, maxx, maxy = roi.region.bounds
    # napari x ∈ [0, 3], napari y ∈ [0, 1]
    # after swap: x = napari_x ∈ [0, 3], y = napari_y ∈ [0, 1]
    assert minx == pytest.approx(0.0)
    assert maxx == pytest.approx(3.0)
    assert miny == pytest.approx(0.0)
    assert maxy == pytest.approx(1.0)


def test_polygon_coords_match_unit_square(square_yx, unit_square_pts):
    """Swapped unit-square data should round-trip back to unit_square_pts."""
    roi = napari_shape_to_roi(square_yx, "polygon")
    # shapely closes the ring, so drop the repeated closing point
    roi_pts = np.array(list(roi.coords))[:-1]
    np.testing.assert_array_almost_equal(
        _sorted_coords(roi_pts),
        _sorted_coords(unit_square_pts),
    )


# ---------------------------------------------------------------------------
# Ellipse
# ---------------------------------------------------------------------------


def test_ellipse_returns_polygon(ellipse_yx):
    roi = napari_shape_to_roi(ellipse_yx, "ellipse", name="my ellipse")
    assert isinstance(roi, PolygonOfInterest)


def test_ellipse_contains_centre(ellipse_yx, ellipse_centre_xy):
    roi = napari_shape_to_roi(ellipse_yx, "ellipse")
    assert roi.contains_point(ellipse_centre_xy)


def test_ellipse_approximate_bounds(
    ellipse_yx, ellipse_centre_xy, ellipse_semi_axes
):
    """Bounding box of the approximated polygon should match the semi-axes."""
    semi_x, semi_y = ellipse_semi_axes
    cx, cy = ellipse_centre_xy
    roi = napari_shape_to_roi(ellipse_yx, "ellipse")
    minx, miny, maxx, maxy = roi.region.bounds
    assert minx == pytest.approx(cx - semi_x, abs=0.01)
    assert maxx == pytest.approx(cx + semi_x, abs=0.01)
    assert miny == pytest.approx(cy - semi_y, abs=0.01)
    assert maxy == pytest.approx(cy + semi_y, abs=0.01)


# ---------------------------------------------------------------------------
# Multi-frame coordinate handling
# ---------------------------------------------------------------------------


def test_three_column_data_strips_frame_index(line_yx, line_xy_expected):
    """3-column (frame, y, x) data should be accepted, frame index stripped."""
    frame_col = np.full((len(line_yx), 1), 5.0)  # all at frame 5
    data_3d = np.hstack([frame_col, line_yx])

    roi = napari_shape_to_roi(data_3d, "line")

    np.testing.assert_array_almost_equal(
        _sorted_coords(roi.coords),
        _sorted_coords(line_xy_expected),
    )


def test_more_than_three_columns_raises(square_yx):
    data_4d = np.ones((len(square_yx), 4))
    with pytest.raises(ValueError, match=re.escape("4 columns")):
        napari_shape_to_roi(data_4d, "polygon")
