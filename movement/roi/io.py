"""I/O functions for saving and loading collections of regions of interest."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import shapely

from movement.roi.line import LineOfInterest
from movement.roi.polygon import PolygonOfInterest
from movement.utils.logging import logger

if TYPE_CHECKING:
    from pathlib import Path

    from movement.roi.base import ROISequence


def save_rois(
    rois: ROISequence,
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

    """
    with open(path) as f:
        data = json.load(f)

    if data["type"] != "FeatureCollection":
        raise logger.error(
            ValueError(f"Expected FeatureCollection, got {data['type']}")
        )

    rois: list[LineOfInterest | PolygonOfInterest] = []
    for feature in data["features"]:
        geometry_data = feature["geometry"]
        properties = feature.get("properties", {})

        geometry = shapely.from_geojson(json.dumps(geometry_data))
        name = properties.get("name")
        roi_type = properties.get("roi_type")

        roi: LineOfInterest | PolygonOfInterest
        if roi_type == "PolygonOfInterest" or isinstance(
            geometry, shapely.Polygon
        ):
            if not isinstance(geometry, shapely.Polygon):
                raise logger.error(
                    TypeError(
                        f"Expected Polygon geometry for PolygonOfInterest, "
                        f"got {type(geometry).__name__}"
                    )
                )
            holes = [interior.coords for interior in geometry.interiors]
            roi = PolygonOfInterest(
                geometry.exterior.coords,
                holes=holes if holes else None,
                name=name,
            )
        elif roi_type == "LineOfInterest" or isinstance(
            geometry, shapely.LineString | shapely.LinearRing
        ):
            if not isinstance(
                geometry, shapely.LineString | shapely.LinearRing
            ):
                raise logger.error(
                    TypeError(
                        f"Expected LineString or LinearRing geometry, "
                        f"got {type(geometry).__name__}"
                    )
                )
            loop = isinstance(geometry, shapely.LinearRing)
            roi = LineOfInterest(geometry.coords, loop=loop, name=name)
        else:
            raise logger.error(
                ValueError(
                    f"Unsupported geometry type: {type(geometry).__name__}"
                )
            )

        rois.append(roi)

    return rois
