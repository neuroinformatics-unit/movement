"""Conversion functions between ``napari`` shapes and ``movement`` RoIs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeAlias

import numpy as np
import shapely
import shapely.affinity

from movement.roi.line import LineOfInterest
from movement.roi.polygon import PolygonOfInterest
from movement.utils.logging import logger

if TYPE_CHECKING:
    from movement.roi.base import BaseRegionOfInterest

NapariShapeType: TypeAlias = Literal[
    "line", "path", "polygon", "rectangle", "ellipse"
]

NAPARI_SHAPE_TO_ROI_CLASS: dict[
    NapariShapeType, type[LineOfInterest] | type[PolygonOfInterest]
] = {
    "line": LineOfInterest,
    "path": LineOfInterest,
    "polygon": PolygonOfInterest,
    "rectangle": PolygonOfInterest,
    "ellipse": PolygonOfInterest,  # approximated as polygon
}


def napari_shape_to_roi(
    data: np.ndarray,
    shape_type: NapariShapeType,
    name: str | None = None,
) -> BaseRegionOfInterest:
    """Convert a 2D ``napari`` shape to a ``movement`` RegionOfInterest (RoI).

    This function only handles static 2D shapes with coordinates (y, x).
    Shapes with additional dimensions will be rejected with an error.

    Parameters
    ----------
    data
        Shape coordinates as stored in ``layer.data[i]``.
        Rows are points; columns are ``(y, x)`` (napari convention).
        A leading frame-index column — i.e. ``(frame, y, x)`` — is stripped
        if present.
    shape_type
        One of the napari shape types, e.g. ``"line"``, ``"polygon"``, etc.
    name
        Name to assign to the resulting RoI. If ``None``, the RoI
        receives the default name defined by
        :class:`~movement.roi.BaseRegionOfInterest`.

    Returns
    -------
    BaseRegionOfInterest
        A :class:`~movement.roi.LineOfInterest` or
        :class:`~movement.roi.PolygonOfInterest`.

    Raises
    ------
    ValueError
        If ``data`` has more than 2 columns (dimensions other than y and x).

    Notes
    -----
    The mapping from napari shape types to movement RoI classes is:

    .. list-table::
       :header-rows: 1

       * - napari shape type
         - movement RoI class
       * - ``"line"``, ``"path"``
         - :class:`~movement.roi.LineOfInterest`
       * - ``"polygon"``, ``"rectangle"``
         - :class:`~movement.roi.PolygonOfInterest`
       * - ``"ellipse"``
         - :class:`~movement.roi.PolygonOfInterest` (approximation)

    Ellipses are approximated as polygons because neither ``movement`` nor
    its underlying geometry library (``shapely``) has a native ellipse type.
    The approximation uses :func:`shapely.Point.buffer` scaled and rotated
    to match the ellipse geometry. This approach was inspired by
    https://gis.stackexchange.com/questions/243459/drawing-ellipse-with-shapely

    """
    data = np.asarray(data, dtype=float)

    # Validate shape only has (y, x) coordinates
    n_cols = data.shape[1]
    if n_cols > 2:
        raise logger.error(
            ValueError(
                f"Shape data has {n_cols} columns, but only 2D shapes with "
                f"coordinates (y, x) are supported."
            )
        )

    # Validate shape_type is recognised
    if shape_type not in NAPARI_SHAPE_TO_ROI_CLASS:
        raise logger.error(
            ValueError(
                f"Unrecognized napari shape type '{shape_type}'. "
                f"Expected one of: {list(NAPARI_SHAPE_TO_ROI_CLASS.keys())}."
            )
        )

    # Swap (y, x) → (x, y) to match movement's coordinate convention
    xy = data[:, ::-1]

    roi_class = NAPARI_SHAPE_TO_ROI_CLASS[shape_type]
    # Approximate ellipses as polygons if needed
    if shape_type == "ellipse":
        xy = _ellipse_to_polygon(xy)
        logger.info(
            f"Ellipse {name or ''} will be approximated as a "
            f"PolygonOfInterest with {xy.shape[0]} vertices."
        )

    return roi_class(xy, name=name)


def _ellipse_to_polygon(xy: np.ndarray) -> np.ndarray:
    """Approximate a napari ellipse as a polygon, returning the vertices.

    napari stores an ellipse as 4 cardinal points. After the (y, x) → (x, y)
    swap has already been applied, these are:

    - ``xy[0]``: one end of the first semi-axis
    - ``xy[1]``: one end of the second semi-axis
    - ``xy[2]``: opposite end of the first semi-axis
    - ``xy[3]``: opposite end of the second semi-axis

    A unit circle is created at the ellipse centre, scaled to the semi-axis
    lengths, and rotated to match the ellipse orientation. The resulting
    polygon's vertices are returned as an (N, 2) array of (x, y),
    without repeating the first vertex at the end.
    """
    centre = (xy[0] + xy[2]) / 2
    axis1 = xy[0] - centre
    axis2 = xy[1] - centre
    semi_a = float(np.linalg.norm(axis1))
    semi_b = float(np.linalg.norm(axis2))
    angle = float(np.degrees(np.arctan2(axis1[1], axis1[0])))

    ellipse_polygon = shapely.affinity.rotate(
        shapely.affinity.scale(
            shapely.Point(centre).buffer(1),
            semi_a,
            semi_b,
        ),
        angle,
    )

    return np.array(ellipse_polygon.exterior.coords)[:-1]
