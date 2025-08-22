"""Utilities for representing and analysing regions of interest."""

from movement.roi.base import BaseRegionOfInterest
from movement.roi.conditions import compute_region_occupancy
from movement.roi.line import LineOfInterest
from movement.roi.polygon import PolygonOfInterest

__all__ = [
    "compute_region_occupancy",
    "LineOfInterest",
    "PolygonOfInterest",
    "BaseRegionOfInterest",
]
