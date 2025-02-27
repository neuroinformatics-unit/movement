import numpy as np
import pytest
import xarray as xr

from movement.roi import PolygonOfInterest, compute_region_occupancy


@pytest.fixture()
def triangle_coords():
    """Coordinates for the right-angled triangle."""
    return [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]


@pytest.fixture()
def triangle(triangle_coords) -> PolygonOfInterest:
    """Triangle."""
    return PolygonOfInterest(triangle_coords, name="triangle")


@pytest.fixture()
def triangle_different_name(triangle_coords) -> PolygonOfInterest:
    """Triangle with a different name."""
    return PolygonOfInterest(triangle_coords, name="pizza_slice")


@pytest.fixture()
def triangle_moved_01(triangle_coords) -> PolygonOfInterest:
    """Triangle moved by 0.01 on the x and y axis."""
    return PolygonOfInterest(
        [(x + 0.01, y + 0.01) for x, y in triangle_coords], name="triangle"
    )


@pytest.fixture()
def triangle_moved_100(triangle_coords) -> PolygonOfInterest:
    """Triangle moved by 1.00 on the x and y axis."""
    return PolygonOfInterest(
        [(x + 1.0, y + 1.0) for x, y in triangle_coords], name="triangle"
    )


@pytest.mark.parametrize(
    "region_fixtures, data, expected_output",
    [
        pytest.param(
            ["triangle", "unit_square", "unit_square_with_hole"],
            np.array([[0.15, 0.15], [0.85, 0.85], [0.5, 0.5], [1.5, 1.5]]),
            {
                "data": np.array(
                    [
                        [True, False, True, False],
                        [True, True, True, False],
                        [True, True, False, False],
                    ]
                ),
                "coords": ["triangle", "Unit square", "Unit square with hole"],
            },
            id="triangle, unit_square, unit_square_with_hole",
        ),
        pytest.param(
            ["triangle", "triangle", "triangle"],
            np.array([[0.15, 0.15], [0.85, 0.85], [0.5, 0.5], [1.5, 1.5]]),
            {
                "data": np.array([[True, False, True, False]] * 3),
                "coords": ["triangle_0", "triangle_1", "triangle_2"],
            },
            id="3 superimposed triangles with same name",
        ),
        pytest.param(
            ["triangle", "triangle_different_name"],
            np.array([[0.5, 0.5]]),
            {
                "data": np.array([True] * 2),
                "coords": ["triangle", "pizza_slice"],
            },
            id="2 superimposed triangles with different names",
        ),
        pytest.param(
            ["triangle", "triangle_moved_01", "triangle_moved_100"],
            np.array([[0.5, 0.5]]),
            {
                "data": np.array([[True], [True], [False]]),
                "coords": ["triangle_0", "triangle_1", "triangle_2"],
            },
            id="3 different triangles with same name",
        ),
        pytest.param(
            ["triangle_different_name"],
            np.array([[0.5, 0.5]]),
            {
                "data": np.array([[True]]),
                "coords": ["pizza_slice"],
            },
            id="1 pizza slice triangle",
        ),
    ],
)
def test_region_occupancy(
    request: pytest.FixtureRequest,
    region_fixtures: list[str],
    data,
    expected_output: dict,
) -> None:
    """Tests region_occupancy for several RoIs.

    Checks whether the dimension, data, and coordinates of the computed
    occupancies are correct.
    """
    regions = [request.getfixturevalue(r) for r in region_fixtures]
    data = xr.DataArray(
        data=data,
        dims=["time", "space"],
        coords={"space": ["x", "y"]},
    )
    occupancies = compute_region_occupancy(data, regions)

    assert occupancies.dims == ("occupancy", "time")
    assert (expected_output["data"] == occupancies.data).all()
    assert occupancies.occupancy.values.tolist() == expected_output["coords"]


def test_region_occupancy_many_regions(
    triangle, unit_square, unit_square_with_hole, triangle_different_name
):
    """Tests occupancy for many RoIs with similar names.

    Ensures correct data and coordinate names for:
        - 1000 triangles suffixed _000 to _999
        - 100 unit squares suffixed _00 to _99
        - 10 unit squares with holes suffixed _0 to _9
        - 1 triangle named "pizza_slice" without suffix
    """
    regions = (
        [triangle] * 1000
        + [unit_square] * 100
        + [unit_square_with_hole] * 10
        + [triangle_different_name] * 1
    )

    data = xr.DataArray(
        data=np.array([[0.5, 0.5]]),
        dims=["time", "space"],
        coords={"space": ["x", "y"]},
    )
    expected_output = {
        "data": np.array([[True]] * 1100 + [[False]] * 10 + [[True]] * 1),
        "coords": (
            [f"triangle_{i:03d}" for i in range(1000)]
            + [f"Unit square_{i:02d}" for i in range(100)]
            + [f"Unit square with hole_{i:01d}" for i in range(10)]
            + ["pizza_slice"]
        ),
    }
    occupancies = compute_region_occupancy(data, regions)

    assert (occupancies.data == expected_output["data"]).all()
    assert occupancies.occupancy.values.tolist() == expected_output["coords"]
