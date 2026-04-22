"""Unit tests for conversion between napari shapes and movement RoIs."""

import numpy as np
import pytest
import shapely
from napari.layers import Shapes

from movement.napari.convert_roi import (
    napari_shape_to_roi,
    napari_shapes_layer_to_rois,
    roi_to_napari_shape,
    rois_to_napari_shapes,
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
    Napari stores ellipses as the 4 corners of the bounding rectangle.
    Bounding box: y=[2, 8], x=[3, 7].
    """
    return np.array([[2.0, 3.0], [2.0, 7.0], [8.0, 7.0], [8.0, 3.0]])


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
    expected_yx = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    np.testing.assert_array_equal(data, expected_yx)


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
# rois_to_napari_shapes
# ===========================================================================


@pytest.mark.parametrize(
    ["roi_fixtures", "expected_shape_types"],
    [
        pytest.param([], [], id="empty sequence"),
        pytest.param(
            ["segment_of_y_equals_x"],
            ["path"],
            id="single LineOfInterest",
        ),
        pytest.param(
            ["unit_square"],
            ["polygon"],
            id="single PolygonOfInterest",
        ),
        pytest.param(
            ["segment_of_y_equals_x", "unit_square", "triangle"],
            ["path", "polygon", "polygon"],
            id="mixed sequence",
        ),
    ],
)
def test_rois_to_napari_shapes_output(
    roi_fixtures, expected_shape_types, request
):
    """Output dict has the correct keys, shape types, and list lengths."""
    rois = [request.getfixturevalue(f) for f in roi_fixtures]
    result = rois_to_napari_shapes(rois)

    assert set(result.keys()) == {"data", "shape_type", "properties"}
    assert result["shape_type"] == expected_shape_types
    assert len(result["data"]) == len(rois)
    assert all(arr.shape[1] == 2 for arr in result["data"])
    assert result["properties"]["name"] == [roi.name for roi in rois]


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
        pytest.param("", "Un-named region", id="empty str uses default name"),
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
    """An ellipse is approximated as a polygon whose bounds match and
    whose area approximates the theoretical ellipse to within 1%.
    """
    roi = napari_shape_to_roi(ellipse_yx, "ellipse")
    # centre (5, 5), semi_x=2, semi_y=3
    # bounds: (5-2, 5-3, 5+2, 5+3)
    assert roi.region.bounds == pytest.approx((3.0, 2.0, 7.0, 8.0), abs=0.01)
    # area: π * semi_x * semi_y = π * 2 * 3 ≈ 18.85
    expected_area = np.pi * 2 * 3
    assert roi.region.area == pytest.approx(expected_area, rel=0.01)


@pytest.mark.parametrize(
    ["data", "shape_type", "match"],
    [
        pytest.param(
            np.array([0.0, 1.0, 2.0]),
            "line",
            "2D array with shape",
            id="1D data raises ValueError",
        ),
        pytest.param(
            np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]),
            "line",
            "2D array with shape",
            id="3-column data raises ValueError",
        ),
        pytest.param(
            np.array([[0.0, 1.0], [2.0, 3.0]]),
            "circle",  # type: ignore[arg-type]
            "Unrecognised napari shape type",
            id="unknown shape_type raises ValueError",
        ),
    ],
)
def test_napari_shape_to_roi_invalid_input(data, shape_type, match):
    """Invalid inputs raise ValueError with an informative message."""
    with pytest.raises(ValueError, match=match):
        napari_shape_to_roi(data, shape_type)


# ===========================================================================
# napari_shapes_layer_to_rois
# ===========================================================================


def test_napari_shapes_layer_to_rois_empty_layer():
    """An empty Shapes layer returns an empty list."""
    rois = napari_shapes_layer_to_rois(Shapes())
    assert rois == []


@pytest.mark.parametrize(
    ["properties", "expected_names"],
    [
        pytest.param(
            {"name": ["sq", "pth"]},
            ["sq", "pth"],
            id="explicit names are preserved",
        ),
        pytest.param(
            {"name": ["sq", ""]},
            ["sq", "Un-named region"],
            id="blank name uses default",
        ),
        pytest.param(
            {},
            ["Un-named region", "Un-named region"],
            id="absent name property uses default",
        ),
    ],
)
def test_napari_shapes_layer_to_rois_output(
    square_yx, path_yx, properties, expected_names
):
    """Returns one RoI per shape with the correct type, name, and order."""
    layer = Shapes(
        [square_yx, path_yx],
        shape_type=["polygon", "path"],
        properties=properties,
    )
    rois = napari_shapes_layer_to_rois(layer)
    assert [type(roi) for roi in rois] == [PolygonOfInterest, LineOfInterest]
    assert [roi.name for roi in rois] == expected_names


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


def test_roundtrip_rois_to_shapes_layer_to_rois(
    segment_of_y_equals_x, unit_square, triangle
):
    """Converting RoIs to a layer and back preserves geometry and names."""
    rois = [segment_of_y_equals_x, unit_square, triangle]
    layer = Shapes(**rois_to_napari_shapes(rois))
    rois2 = napari_shapes_layer_to_rois(layer)
    assert len(rois2) == len(rois)
    for roi, roi2 in zip(rois, rois2, strict=True):
        assert shapely.normalize(roi.region) == shapely.normalize(roi2.region)
        assert roi.name == roi2.name
