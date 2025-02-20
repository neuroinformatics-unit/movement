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
def test_region_occupancy(
    triangle: PolygonOfInterest,
    unit_square: PolygonOfInterest,
    unit_square_with_hole: PolygonOfInterest,
) -> None:
    data = xr.DataArray(
        data=np.array([[0.15, 0.15], [0.85, 0.85], [0.5, 0.5], [1.5, 1.5]]),
        dims=["time", "space"],
        coords={"space": ["x", "y"]},
    )
    # NB results needs to be (occupancy, time)!
    results = np.array(
        [
            [True, True, True, False],
            [True, False, True, False],
            [True, True, False, False],
        ]
    )
    occupancies = compute_region_occupancy(
        data, [unit_square, triangle, unit_square_with_hole]
    )

    assert (results == occupancies.values).all()
