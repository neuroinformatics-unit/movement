"""I/O functions for saving and loading collections of regions of interest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import shapely

from movement.roi.line import LineOfInterest
from movement.roi.polygon import PolygonOfInterest
from movement.validators.files import ValidROICollectionGeoJSON

if TYPE_CHECKING:
    from movement.roi.base import ROICollection


def save_rois(
    rois: ROICollection,
    path: str | Path,
) -> None:
    """Save a collection of regions of interest to a GeoJSON file.

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
    BaseRegionOfInterest.to_file : Save a single ROI to a file.

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

    with open(path, "w") as f:
        json.dump(feature_collection, f, indent=2)


def load_rois(path: str | Path) -> list[LineOfInterest | PolygonOfInterest]:
    """Load a collection of regions of interest from a GeoJSON file.

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
    LineOfInterest.from_file : Load a single LineOfInterest from a file.
    PolygonOfInterest.from_file : Load a single PolygonOfInterest from a file.

    Examples
    --------
    Load a collection of ROIs from a GeoJSON file:

    >>> from movement.roi import load_rois
    >>> rois = load_rois("/path/to/rois.geojson")

    This returns a list of ROI objects, we can check their names:

    >>> [roi.name for roi in rois]
    ['square', 'diagonal']

    """
    validated_file = ValidROICollectionGeoJSON(Path(path))
    with open(validated_file.path) as f:
        data = json.load(f)

    return [_feature_to_roi(feature) for feature in data["features"]]


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
