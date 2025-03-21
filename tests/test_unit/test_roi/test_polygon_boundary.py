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
                np.array([[0.0, 0.0], [0.0, 0.25], [0.25, 0.0]]),
                np.array([[0.75, 0.0], [1.0, 0.25], [1.0, 0.0]]),
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
    expected_holes = tuple(shapely.Polygon(ib) for ib in interior_boundaries)

    computed_exterior = polygon.exterior_boundary
    computed_interiors = polygon.interior_boundaries
    computed_holes = polygon.holes

    assert isinstance(computed_exterior, LineOfInterest)
    assert expected_exterior.equals_exact(computed_exterior.region, tolerance)
    assert isinstance(computed_interiors, tuple)
    assert isinstance(computed_holes, tuple)
    assert len(computed_interiors) == len(expected_interiors)
    assert len(computed_holes) == len(expected_holes)
    assert len(computed_holes) == len(computed_interiors)
    for i, interior_line in enumerate(computed_interiors):
        assert isinstance(interior_line, LineOfInterest)

        assert expected_interiors[i].equals_exact(
            interior_line.region, tolerance
        )
    for i, interior_hole in enumerate(computed_holes):
        assert isinstance(interior_hole, PolygonOfInterest)

        assert expected_holes[i].equals_exact(interior_hole.region, tolerance)
