from typing import Any

import numpy as np
import pytest
import xarray as xr

from movement.roi.base import BaseRegionOfInterest
from movement.roi.polygon import PolygonOfInterest


def walk_inside_unit_square() -> np.ndarray:
    """Return an array representing a walk around the unit square's interior.

    Note that this walk does not fall into the "hole" introduced by the
    `unit_square_with_hole` fixture.
    """
    return np.array(
        [
            [0.05, 0.05],
            [0.50, 0.05],
            [0.95, 0.05],
            [0.95, 0.50],
            [0.95, 0.95],
            [0.50, 0.95],
            [0.05, 0.95],
            [0.05, 0.50],
        ]
    )


def walk_from_left_to_right() -> np.ndarray:
    """Return an array representing a walk through the unit square.

    The walk starts at (-0.05, 0.05) and ends at (1.05, 0.95),
    proceeding in a diagonal line.
    """
    x_points = np.linspace(-0.05, 1.05, num=5, endpoint=True)
    y_points = np.linspace(0.05, 0.95, num=5, endpoint=True)
    return np.array([x_points, y_points]).transpose()


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


@pytest.mark.parametrize(
    ["region", "data", "fn_kwargs", "expected_result"],
    [
        pytest.param(
            "unit_square",
            xr.DataArray(
                data=walk_inside_unit_square(),
                dims=["time", "space"],
                coords={"space": ["x", "y"]},
            ),
            {},
            [True] * walk_inside_unit_square().shape[0],
            id="Inside unit square",
        ),
        pytest.param(
            "unit_square_with_hole",
            xr.DataArray(
                data=walk_inside_unit_square(),
                dims=["time", "space"],
                coords={"space": ["x", "y"]},
            ),
            {},
            [True] * walk_inside_unit_square().shape[0],
            id="Inside unit square, avoiding the hole",
        ),
        pytest.param(
            "unit_square",
            xr.DataArray(
                data=walk_from_left_to_right(),
                dims=["time", "space"],
                coords={"space": ["x", "y"]},
            ),
            {},
            [False, True, True, True, False],
            id="Across the unit square",
        ),
        pytest.param(
            "unit_square_with_hole",
            xr.DataArray(
                data=walk_from_left_to_right(),
                dims=["time", "space"],
                coords={"space": ["x", "y"]},
            ),
            {},
            [False, True, False, True, False],
            id="Across the unit square, fall into the hole",
        ),
        pytest.param(
            "unit_square",
            xr.DataArray(
                data=np.array([[0.0, 0.0], [1.0, 0.0]]),
                dims=["time", "space"],
                coords={"space": ["x", "y"]},
            ),
            {},
            [True, True],
            id="Boundary is included by default",
        ),
        pytest.param(
            "unit_square",
            xr.DataArray(
                data=np.array([[0.0, 0.0], [1.0, 0.0]]),
                dims=["time", "space"],
                coords={"space": ["x", "y"]},
            ),
            {"include_boundary": False},
            [False, False],
            id="Boundary can be ignored",
        ),
    ],
)
def test_is_inside(
    region: BaseRegionOfInterest,
    data: xr.DataArray,
    fn_kwargs: dict[str, Any],
    expected_result: np.ndarray,
    request,
) -> None:
    if isinstance(region, str):
        region = request.getfixturevalue(region)
    if not isinstance(expected_result, np.ndarray):
        expected_result = np.array(expected_result)

    result = region.is_inside(data, **fn_kwargs)

    assert isinstance(result, xr.DataArray)
    assert result.shape == expected_result.shape
    assert (result.values == expected_result).all()
