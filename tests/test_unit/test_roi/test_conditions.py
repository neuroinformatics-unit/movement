import numpy as np
import pytest
import xarray as xr

from movement.roi import PolygonOfInterest, compute_region_occupancy


@pytest.fixture()
def triangle() -> PolygonOfInterest:
    return PolygonOfInterest(
        [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)], name="triangle"
    )


# TSTK need to parametrise this!
# 1) There is no check that RoI names match the coordinates along the created
#   dimension
# 2) There is no check that duplicate names are handled in the expected way
#   (appends _00, 01, 02, ... 09, 10, etc)
# Need to reorganise the inputs; should probably be
# list_of_regions (or list_of_strs_of_region_fixture_names)
# expected_output_array (np.array of bools, like ``results`` below)
# expected_output_coordinates (list of strings, the coordinates along the
#   output occupancy dimension)


@pytest.mark.parametrize(
    "region_fixtures, data, results",
    [
        pytest.param(
            ["triangle", "unit_square", "unit_square_with_hole"],
            np.array([[0.15, 0.15], [0.85, 0.85], [0.5, 0.5], [1.5, 1.5]]),
            np.array(
                [
                    [True, False, True, False],
                    [True, True, True, False],
                    [True, True, False, False],
                ]
            ),
            id="triangle, unit_square, unit_square_with_hole",
        )
    ],
)
def test_region_occupancy(
    request,
    region_fixtures,
    data,
    results,
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
    assert (results == occupancies.values).all()
