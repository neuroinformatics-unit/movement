"""Unit tests for conversion between napari shapes and movement RoIs."""

import numpy as np
import pytest
import shapely

from movement.napari.convert_roi import (
    napari_shape_to_roi,
    roi_to_napari_shape,
)
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
    """Non-square rectangle in napari (y, x) convention."""
    return np.array([[0.0, 0.0], [0.0, 3.0], [1.0, 3.0], [1.0, 0.0]])


@pytest.fixture
def ellipse_yx():
    """Axis-aligned ellipse in napari (y, x) convention.

    Centre (cy=5, cx=5), semi-axes ry=3, rx=2.
    The 4 cardinal points are: top, right, bottom, left.
    """
    return np.array([[2.0, 5.0], [5.0, 7.0], [8.0, 5.0], [5.0, 3.0]])


# ===========================================================================
# roi_to_napari_shape
# ===========================================================================


@pytest.mark.parametrize(
    ["roi_fixture", "expected_shape_type"],
    [
        pytest.param(
            "segment_of_y_equals_x", "path", id="LineOfInterest → path"
        ),
        pytest.param(
            "unit_square", "polygon", id="PolygonOfInterest → polygon"
        ),
    ],
)
def test_roi_to_napari_shape_type(roi_fixture, expected_shape_type, request):
    """Each RoI class maps to the correct napari shape type."""
    roi = request.getfixturevalue(roi_fixture)
    _, shape_type = roi_to_napari_shape(roi)
    assert shape_type == expected_shape_type


@pytest.mark.parametrize(
    "roi_fixture",
    [
        pytest.param("segment_of_y_equals_x", id="LineOfInterest"),
        pytest.param("unit_square", id="PolygonOfInterest (square)"),
        pytest.param("triangle", id="PolygonOfInterest (triangle)"),
    ],
)
def test_roi_to_napari_shape_output_array(roi_fixture, request):
    """Output is an (N, 2) array with no repeated closing vertex."""
    roi = request.getfixturevalue(roi_fixture)
    data, _ = roi_to_napari_shape(roi)
    assert data.ndim == 2
    assert data.shape[1] == 2
    assert not np.array_equal(data[0], data[-1])


def test_roi_to_napari_shape_coordinate_swap(unit_square):
    """Coordinates are returned in (y, x) order (swapped from (x, y))."""
    data, _ = roi_to_napari_shape(unit_square)
    # coords property returns (x, y); strip the shapely closing vertex
    expected_xy = np.array(unit_square.coords)[:-1]
    np.testing.assert_array_equal(data, expected_xy[:, ::-1])


def test_roi_to_napari_shape_closed_line_warning(caplog):
    """Converting a closed LineOfInterest emits a warning about the lost
    closing segment.
    """
    closed_line = LineOfInterest([(0, 0), (1, 0), (0, 1)], loop=True)
    data, shape_type = roi_to_napari_shape(closed_line)
    assert shape_type == "path"
    assert len(data) == 3  # closing vertex stripped, 3 unique points remain
    assert any("closing segment" in msg for msg in caplog.messages)


# ===========================================================================
# napari_shape_to_roi
# ===========================================================================


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
def test_napari_shape_to_roi_type(
    shape_type, data_fixture, expected_type, request
):
    """Each napari shape type maps to the correct movement RoI class."""
    data = request.getfixturevalue(data_fixture)
    assert isinstance(napari_shape_to_roi(data, shape_type), expected_type)


@pytest.mark.parametrize(
    ["name", "expected_name"],
    [
        pytest.param("my boundary", "my boundary", id="explicit name"),
        pytest.param(None, "Un-named region", id="None uses default name"),
    ],
)
def test_napari_shape_to_roi_name(line_yx, name, expected_name):
    """The ``name`` argument is assigned to the resulting RoI."""
    roi = napari_shape_to_roi(line_yx, "line", name=name)
    assert roi.name == expected_name


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
            id="polygon: (y,x) coords are swapped to (x,y)",
        ),
        pytest.param(
            "rectangle",
            "nonsquare_rect_yx",
            shapely.Polygon([[0.0, 0.0], [3.0, 0.0], [3.0, 1.0], [0.0, 1.0]]),
            id="rectangle: (y,x) coords are swapped to (x,y)",
        ),
    ],
)
def test_napari_shape_to_roi_coordinate_swap(
    shape_type, data_fixture, expected_geometry, request
):
    """Napari (y, x) coordinates are converted to movement (x, y)."""
    data = request.getfixturevalue(data_fixture)
    roi = napari_shape_to_roi(data, shape_type)
    assert shapely.normalize(roi.region) == shapely.normalize(
        expected_geometry
    )


def test_napari_shape_to_roi_ellipse_approximation(ellipse_yx):
    """An ellipse is approximated as a polygon whose bounds match the
    theoretical semi-axes to within 1% of the ellipse dimensions.
    """
    roi = napari_shape_to_roi(ellipse_yx, "ellipse")
    # centre (5, 5), semi_x=2, semi_y=3 → (5-2, 5-3, 5+2, 5+3)
    assert roi.region.bounds == pytest.approx((3.0, 2.0, 7.0, 8.0), abs=0.01)


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
def test_napari_shape_to_roi_invalid_input(data, shape_type, match):
    """Invalid inputs raise ValueError with an informative message."""
    with pytest.raises(ValueError, match=match):
        napari_shape_to_roi(data, shape_type)


# ===========================================================================
# Roundtrips
# ===========================================================================


@pytest.mark.parametrize(
    "roi_fixture",
    [
        pytest.param("segment_of_y_equals_x", id="LineOfInterest"),
        pytest.param("unit_square", id="PolygonOfInterest (square)"),
        pytest.param("triangle", id="PolygonOfInterest (triangle)"),
    ],
)
def test_roundtrip_roi_to_napari_to_roi(roi_fixture, request):
    """Converting a RoI to a napari shape and back preserves the geometry."""
    roi = request.getfixturevalue(roi_fixture)
    data, shape_type = roi_to_napari_shape(roi)
    roi2 = napari_shape_to_roi(data, shape_type)
    assert shapely.normalize(roi.region) == shapely.normalize(roi2.region)


@pytest.mark.parametrize(
    ["shape_type", "data_fixture", "expected_shape_type_back"],
    [
        pytest.param("line", "line_yx", "path", id="line → path"),
        pytest.param("path", "path_yx", "path", id="path → path"),
        pytest.param(
            "polygon", "square_yx", "polygon", id="polygon → polygon"
        ),
        pytest.param(
            "rectangle", "square_yx", "polygon", id="rectangle → polygon"
        ),
        pytest.param(
            "ellipse", "ellipse_yx", "polygon", id="ellipse → polygon"
        ),
    ],
)
def test_roundtrip_napari_to_roi_to_napari(
    shape_type, data_fixture, expected_shape_type_back, request
):
    """Converting a napari shape to a RoI and back gives a predictable shape
    type: ``"line"`` and ``"path"`` both return as ``"path"``;
    ``"polygon"``, ``"rectangle"``, and ``"ellipse"`` all return as
    ``"polygon"``.
    """
    data = request.getfixturevalue(data_fixture)
    roi = napari_shape_to_roi(data, shape_type)
    _, shape_type_back = roi_to_napari_shape(roi)
    assert shape_type_back == expected_shape_type_back
