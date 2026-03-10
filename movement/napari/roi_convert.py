"""Conversion functions between napari shapes and movement RoI objects."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import shapely
import shapely.affinity

from movement.roi.line import LineOfInterest
from movement.roi.polygon import PolygonOfInterest
from movement.utils.logging import logger

if TYPE_CHECKING:
    from movement.roi.base import BaseRegionOfInterest

NapariShapeType = Literal["line", "path", "polygon", "rectangle", "ellipse"]

_NAPARI_SHAPE_TO_ROI_CLASS: dict[
    str, type[LineOfInterest] | type[PolygonOfInterest]
] = {
    "line": LineOfInterest,
    "path": LineOfInterest,
    "polygon": PolygonOfInterest,
    "rectangle": PolygonOfInterest,
}


def napari_shape_to_roi(
    data: np.ndarray,
    shape_type: NapariShapeType,
    name: str | None = None,
) -> BaseRegionOfInterest:
    """Convert a single napari shape to a movement RegionOfInterest (RoI).

    Parameters
    ----------
    data
        Shape coordinates as stored in ``layer.data[i]``.
        Rows are points; columns are ``(y, x)`` (napari convention).
        A leading frame-index column — i.e. ``(frame, y, x)`` — is stripped
        (an info message is logged).
    shape_type
        One of the napari shape types.
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
        If ``data`` has more than 3 columns (more than one leading
        non-spatial dimension), which is not supported.

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
    to match the ellipse geometry.

    """
    data = np.asarray(data, dtype=float)

    n_cols = data.shape[1]
    if n_cols > 3:
        raise logger.error(
            ValueError(
                f"Shape data has {n_cols} columns, but only 2D shapes with "
                f"coordinates (y, x) or (frame, y, x) are supported."
            )
        )
    if n_cols == 3:
        logger.info(
            f"Shape '{name or 'Un-named'}' has a leading frame-index column "
            f"(frame, y, x). The frame index will be stripped and the "
            f"shape will be treated as a static spatial RoI."
        )
        data = data[:, 1:]  # drop frame column, keep (y, x)

    # Swap (y, x) → (x, y) to match movement's coordinate convention
    xy = data[:, ::-1]

    roi_class = _NAPARI_SHAPE_TO_ROI_CLASS.get(shape_type)
    if roi_class is not None:
        return roi_class(xy, name=name)

    return _ellipse_to_polygon(xy, name=name)


def _ellipse_to_polygon(
    xy: np.ndarray,
    name: str | None,
) -> PolygonOfInterest:
    """Approximate a napari ellipse as a PolygonOfInterest.

    napari stores an ellipse as 4 cardinal points. After the (y, x) → (x, y)
    swap has already been applied, these are:

    - ``xy[0]``: one end of the first semi-axis
    - ``xy[1]``: one end of the second semi-axis
    - ``xy[2]``: opposite end of the first semi-axis
    - ``xy[3]``: opposite end of the second semi-axis

    A unit circle is created at the ellipse centre, scaled to the semi-axis
    lengths, and rotated to match the ellipse orientation.
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

    n_vertices = len(ellipse_polygon.exterior.coords) - 1
    logger.info(
        f"Ellipse '{name or 'Un-named'}' will be approximated as a "
        f"PolygonOfInterest with {n_vertices} vertices."
    )
    return PolygonOfInterest(ellipse_polygon.exterior.coords, name=name)
