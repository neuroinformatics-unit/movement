"""I/O functions for saving and loading collections of regions of interest."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, TypeAlias

import shapely

from movement.roi.base import BaseRegionOfInterest
from movement.roi.line import LineOfInterest
from movement.roi.polygon import PolygonOfInterest
from movement.validators.files import (
    ValidROICollectionGeoJSON,
    validate_file_path,
)

if TYPE_CHECKING:
    from pathlib import Path


# Type alias for collections of ROIs
ROICollection: TypeAlias = Sequence[BaseRegionOfInterest]


def save_rois(
    rois: ROICollection,
    path: str | Path,
) -> None:
    """Save a collection of regions of interest (ROIs) to a GeoJSON file.

    The ROIs are saved as a GeoJSON FeatureCollection, with each ROI
    represented as a Feature containing its geometry and properties
    (name and ROI type).

    Parameters
    ----------
    rois : list of BaseRegionOfInterest
        The regions of interest to save.
    path : str or pathlib.Path
        Path to the output file. Should have a ``.geojson`` extension.

    See Also
    --------
    load_rois : Load a collection of ROIs from a GeoJSON file.

    Examples
    --------
    Create a polygon and a line, then save them to a GeoJSON file:

    >>> from movement.roi import LineOfInterest, PolygonOfInterest, save_rois
    >>> square = PolygonOfInterest(
    ...     [(0, 0), (1, 0), (1, 1), (0, 1)], name="square"
    ... )
    >>> diagonal = LineOfInterest([(0, 0), (1, 1)], name="diagonal")
    >>> save_rois([square, diagonal], "/path/to/rois.geojson")

    """
    valid_path = validate_file_path(
        path, permission="w", suffixes=ValidROICollectionGeoJSON.suffixes
    )

    features = []
    for roi in rois:
        geometry = json.loads(shapely.to_geojson(roi.region))
        feature = {
            "type": "Feature",
            "geometry": geometry,
            "properties": {
                "name": roi.name,
                "roi_type": roi.__class__.__name__,
            },
        }
        features.append(feature)

    feature_collection = {
        "type": "FeatureCollection",
        "features": features,
    }

    with open(valid_path, "w") as f:
        json.dump(feature_collection, f, indent=2)


def load_rois(path: str | Path) -> list[LineOfInterest | PolygonOfInterest]:
    """Load a collection of regions of interest (ROIs) from a GeoJSON file.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the GeoJSON file to load.

    Returns
    -------
    list of LineOfInterest or PolygonOfInterest
        The loaded regions of interest.

    See Also
    --------
    save_rois : Save a collection of ROIs to a GeoJSON file.

    Examples
    --------
    Load a collection of ROIs from a GeoJSON file:

    >>> from movement.roi import load_rois
    >>> rois = load_rois("/path/to/rois.geojson")

    This returns a list of ROI objects, we can check their names:

    >>> [roi.name for roi in rois]
    ['square', 'diagonal']

    """
    validated_file = ValidROICollectionGeoJSON(path)
    return [
        _feature_to_roi(feature)
        for feature in validated_file.data["features"]
    ]


def _feature_to_roi(
    feature: dict[str, Any],
) -> LineOfInterest | PolygonOfInterest:
    """Convert a validated GeoJSON feature to an ROI object."""
    geometry_data = feature["geometry"]
    properties = feature.get("properties") or {}

    geometry = shapely.from_geojson(json.dumps(geometry_data))
    name = properties.get("name")

    if isinstance(geometry, shapely.Polygon):
        holes = [interior.coords for interior in geometry.interiors]
        return PolygonOfInterest(
            geometry.exterior.coords,
            holes=holes if holes else None,
            name=name,
        )
    else:  # Must be LineString or LinearRing (validator ensures that)
        loop = isinstance(geometry, shapely.LinearRing)
        return LineOfInterest(geometry.coords, loop=loop, name=name)
