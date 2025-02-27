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
                "data": np.array(
                    [
                        [True, False, True, False],
                        [True, False, True, False],
                        [True, False, True, False],
                    ]
                ),
                "coords": ["triangle_0", "triangle_1", "triangle_2"],
            },
            id="3 superimposed triangles with same name",
        ),
        pytest.param(
            ["triangle", "triangle_different_name"],
            np.array([[0.15, 0.15], [0.85, 0.85], [0.5, 0.5], [1.5, 1.5]]),
            {
                "data": np.array(
                    [
                        [True, False, True, False],
                        [True, False, True, False],
                    ]
                ),
                "coords": ["triangle", "pizza_slice"],
            },
            id="2 superimposed triangles with different names",
        ),
        pytest.param(
            ["triangle", "triangle_moved_01"],
            np.array([[0.15, 0.15], [0.85, 0.85], [0.5, 0.5], [1.5, 1.5]]),
            {
                "data": np.array(
                    [
                        [True, False, True, False],
                        [True, False, True, False],
                    ]
                ),
                "coords": ["triangle_0", "triangle_1"],
            },
            id="2 different triangles with same name",
        ),
        pytest.param(
            ["triangle", "triangle_moved_01", "triangle_moved_100"],
            np.array([[0.15, 0.15], [0.85, 0.85], [0.5, 0.5], [1.5, 1.5]]),
            {
                "data": np.array(
                    [
                        [True, False, True, False],
                        [True, False, True, False],
                        [False, False, False, True],
                    ]
                ),
                "coords": ["triangle_0", "triangle_1", "triangle_2"],
            },
            id="3 different triangles with same name",
        ),
        pytest.param(
            ["triangle", "unit_square"],
            np.array([[0.15, 0.15], [0.5, 0.5]]),
            {
                "data": np.array(
                    [
                        [True, True],
                        [True, True],
                    ]
                ),
                "coords": ["triangle", "Unit square"],
            },
            id="triangle, square, data points occupy both regions",
        ),
    ],
)
def test_region_occupancy(
    request: pytest.FixtureRequest,
    region_fixtures: list[str],
    data,
    expected_output: dict,
) -> None:
    """Tests region_occupancy for several RoIs."""
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


def test_region_occupancy_1000_triangles(triangle):
    """Tests region_occupancy 1000 triangles with the same name."""
    regions = [triangle] * 1000
    data = xr.DataArray(
        data=np.array([[0.15, 0.15], [0.85, 0.85], [0.5, 0.5], [1.5, 1.5]]),
        dims=["time", "space"],
        coords={"space": ["x", "y"]},
    )
    expected_output = {
        "data": np.array([[True, False, True, False]] * 1000),
        "coords": [f"triangle_{i:03d}" for i in range(1000)],
    }
    occupancies = compute_region_occupancy(data, regions)

    assert occupancies.dims == ("occupancy", "time")
    assert (expected_output["data"] == occupancies.data).all()
    assert occupancies.occupancy.values.tolist() == expected_output["coords"]
