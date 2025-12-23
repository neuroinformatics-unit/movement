import re
from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
import shapely

from movement.roi.line import LineOfInterest
from movement.roi.polygon import PolygonOfInterest


@pytest.mark.parametrize(
    ["exterior", "holes", "name", "expected"],
    [
        pytest.param(
            "unit_square_pts",
            None,
            None,
            does_not_raise("Un-named region"),
            id="Polygon default name",
        ),
        pytest.param(
            "unit_square_pts",
            "unit_square_hole",
            "elephant",
            does_not_raise("elephant"),
            id="Polygon explicit name",
        ),
        pytest.param(
            np.array([[0.0, 0.0], [1.0, 0.0]]),
            None,
            None,
            pytest.raises(
                ValueError,
                match=re.escape(
                    "Need at least 3 points to define a 2D region (got 2)"
                ),
            ),
            id="Polygon too few points",
        ),
    ],
)
def test_polygon_creation(exterior, holes, name, expected, request):
    exterior = (
        request.getfixturevalue(exterior)
        if isinstance(exterior, str)
        else exterior
    )
    holes_val = (
        [request.getfixturevalue(holes)] if isinstance(holes, str) else None
    )
    with expected as expected_name:
        roi = PolygonOfInterest(exterior, holes=holes_val, name=name)
        assert isinstance(roi.region, shapely.Polygon)
        assert roi.is_closed
        assert roi.dimensions == 2
        assert len(roi.coords) == len(exterior) + 1
        assert roi.name == expected_name
        assert repr(roi) == str(roi)
        assert "-gon" in repr(roi)


@pytest.mark.parametrize(
    ["points", "loop", "name", "expected"],
    [
        pytest.param(
            np.array([[0.0, 0.0], [1.0, 0.0]]),
            False,
            None,
            does_not_raise(
                {
                    "is_closed": False,
                    "dimensions": 1,
                    "coords_offset": 0,
                }
            ),
            id="Line open",
        ),
        pytest.param(
            np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]),
            True,
            "triangle",
            does_not_raise(
                {
                    "name": "triangle",
                    "is_closed": True,
                    "dimensions": 1,
                    "coords_offset": 1,
                }
            ),
            id="Line looped",
        ),
        pytest.param(
            np.array([[0.0, 0.0]]),
            False,
            None,
            pytest.raises(
                ValueError,
                match=re.escape(
                    "Need at least 2 points to define a 1D region (got 1)"
                ),
            ),
            id="Line too few points",
        ),
        pytest.param(
            np.array([[0.0, 0.0], [1.0, 0.0]]),
            True,
            None,
            pytest.raises(
                ValueError,
                match=re.escape(
                    "Cannot create a loop from a single line segment."
                ),
            ),
            id="Line loop invalid",
        ),
    ],
)
def test_line_creation(points, loop, name, expected, request):
    with expected as expected:
        roi = LineOfInterest(points, loop=loop, name=name)
        assert roi.name == expected.get("name", "Un-named region")
        assert roi.dimensions == expected["dimensions"]
        assert roi.is_closed == expected["is_closed"]
        # LinearRing closes the path, so add 1 when looped
        assert len(roi.coords) == len(points) + expected["coords_offset"]
        assert repr(roi) == str(roi)
        assert "line segment(s)" in repr(roi)
