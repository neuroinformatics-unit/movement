import re
from typing import Any

import numpy as np
import pytest
import shapely

from movement.roi.base import BaseRegionOfInterest


@pytest.mark.parametrize(
    ["input_pts", "kwargs_for_creation", "expected_results"],
    [
        pytest.param(
            "unit_square_pts",
            {"dimensions": 2, "closed": False},
            {"is_closed": True, "dimensions": 2, "name": "Un-named region"},
            id="Polygon, closed is ignored",
        ),
        pytest.param(
            "unit_square_pts",
            {"dimensions": 1, "closed": False},
            {"is_closed": False, "dimensions": 1},
            id="Line segment(s)",
        ),
        pytest.param(
            "unit_square_pts",
            {"dimensions": 1, "closed": True},
            {"is_closed": True, "dimensions": 1},
            id="Looped lines",
        ),
        pytest.param(
            "unit_square_pts",
            {"dimensions": 2, "name": "elephant"},
            {"is_closed": True, "dimensions": 2, "name": "elephant"},
            id="Explicit name",
        ),
        pytest.param(
            np.array([[0.0, 0.0], [1.0, 0.0]]),
            {"dimensions": 2},
            ValueError("Need at least 3 points to define a 2D region (got 2)"),
            id="Too few points (2D)",
        ),
        pytest.param(
            np.array([[0.0, 0.0]]),
            {"dimensions": 1},
            ValueError("Need at least 2 points to define a 1D region (got 1)"),
            id="Too few points (1D)",
        ),
        pytest.param(
            np.array([[0.0, 0.0], [1.0, 0.0]]),
            {"dimensions": 1},
            {"is_closed": False},
            id="Borderline enough points (1D)",
        ),
        pytest.param(
            np.array([[0.0, 0.0], [1.0, 0.0]]),
            {"dimensions": 1, "closed": True},
            ValueError("Cannot create a loop from a single line segment."),
            id="Cannot close single line segment.",
        ),
        pytest.param(
            "unit_square_pts",
            {"dimensions": 3, "closed": False},
            ValueError(
                "Only regions of interest of dimension 1 or 2 "
                "are supported (requested 3)"
            ),
            id="Bad dimensionality",
        ),
    ],
)
def test_creation(
    input_pts,
    kwargs_for_creation: dict[str, Any],
    expected_results: dict[str, Any] | Exception,
    request,
) -> None:
    if isinstance(input_pts, str):
        input_pts = request.getfixturevalue(input_pts)

    if isinstance(expected_results, Exception):
        with pytest.raises(
            type(expected_results), match=re.escape(str(expected_results))
        ):
            BaseRegionOfInterest(input_pts, **kwargs_for_creation)
    else:
        roi = BaseRegionOfInterest(input_pts, **kwargs_for_creation)

        expected_dim = kwargs_for_creation.pop("dimensions", 2)
        expected_closure = kwargs_for_creation.pop("closed", False)
        if expected_dim == 2:
            assert isinstance(roi.region, shapely.Polygon)
            assert len(roi.coords) == len(input_pts) + 1
            string_should_contain = "-gon"
        elif expected_closure:
            assert isinstance(roi.region, shapely.LinearRing)
            assert len(roi.coords) == len(input_pts) + 1
            string_should_contain = "line segment(s)"
        else:
            assert isinstance(roi.region, shapely.LineString)
            assert len(roi.coords) == len(input_pts)
            string_should_contain = "line segment(s)"
        assert string_should_contain in roi.__str__()
        assert string_should_contain in roi.__repr__()

        for attribute_name, expected_value in expected_results.items():
            assert getattr(roi, attribute_name) == expected_value
