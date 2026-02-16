"""Utilities for representing and analysing regions of interest."""

from movement.roi.base import BaseRegionOfInterest
from movement.roi.conditions import compute_region_occupancy
from movement.roi.io import load_rois, save_rois, ROICollection
from movement.roi.line import LineOfInterest
from movement.roi.polygon import PolygonOfInterest

__all__ = [
    "BaseRegionOfInterest",
    "LineOfInterest",
    "PolygonOfInterest",
    "ROICollection",
    "compute_region_occupancy",
    "load_rois",
    "save_rois",
]
