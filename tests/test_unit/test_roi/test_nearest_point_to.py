import numpy as np

from movement.roi.base import BaseRegionOfInterest


def test_nearest_point_to(unit_square: BaseRegionOfInterest) -> None:
    nearest = unit_square.nearest_point_to(np.array([-1.0, 0.0]))

    pass
