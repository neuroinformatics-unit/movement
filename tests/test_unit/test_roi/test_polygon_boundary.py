import numpy as np
import pytest
import shapely

from movement.roi.line import LineOfInterest
from movement.roi.polygon import PolygonOfInterest


@pytest.mark.parametrize(
    ["exterior_boundary", "interior_boundaries"],
    [
        pytest.param("unit_square_pts", tuple(), id="No holes"),
        pytest.param(
            "unit_square_pts", tuple(["unit_square_hole"]), id="One hole"
        ),
        pytest.param(
            "unit_square_pts",
            (
                np.array([[0.0, 0.0], [0.25, 0.0], [0.0, 0.25]]),
                np.array([[0.75, 0.0], [1.0, 0.0], [1.0, 0.25]]),
            ),
            id="Corners shaved off",
        ),
    ],
)
def test_boundary(exterior_boundary, interior_boundaries, request) -> None:
    if isinstance(exterior_boundary, str):
        exterior_boundary = request.getfixturevalue(exterior_boundary)
    interior_boundaries = tuple(
        request.getfixturevalue(ib) if isinstance(ib, str) else ib
        for ib in interior_boundaries
    )
    tolerance = 1.0e-8

    polygon = PolygonOfInterest(
        exterior_boundary, holes=interior_boundaries, name="Holey"
    )
    expected_exterior = shapely.LinearRing(exterior_boundary)
    expected_interiors = tuple(
        shapely.LinearRing(ib) for ib in interior_boundaries
    )

    computed_exterior = polygon.exterior_boundary
    computed_interiors = polygon.interior_boundaries

    assert isinstance(computed_exterior, LineOfInterest)
    assert expected_exterior.equals_exact(computed_exterior.region, tolerance)
    assert isinstance(computed_interiors, tuple)
    assert len(computed_interiors) == len(expected_interiors)
    for i, item in enumerate(computed_interiors):
        assert isinstance(item, LineOfInterest)

        assert expected_interiors[i].equals_exact(item.region, tolerance)
