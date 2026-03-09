"""Conversion functions between napari shapes and movement RoI objects."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal

import numpy as np

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
    ellipse_n_vertices: int = 64,
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
    ellipse_n_vertices
        Number of polygon vertices used to approximate ellipses.
        Default ``64``.

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
    A :class:`UserWarning` is emitted whenever an ellipse is converted.
    The number of vertices in the approximation is controlled by
    ``ellipse_n_vertices``; the default of ``64`` is sufficient for most
    practical purposes at typical image resolutions.

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

    return _ellipse_to_roi(xy, name=name, n_vertices=ellipse_n_vertices)


def _ellipse_to_roi(
    xy: np.ndarray,
    name: str | None,
    n_vertices: int,
) -> PolygonOfInterest:
    """Approximate a napari ellipse as a PolygonOfInterest.

    napari stores an ellipse as 4 cardinal points. After the (y, x) → (x, y)
    swap has already been applied, these are:

    - ``xy[0]``: one end of the first semi-axis
    - ``xy[1]``: one end of the second semi-axis
    - ``xy[2]``: opposite end of the first semi-axis
    - ``xy[3]``: opposite end of the second semi-axis

    The ellipse is parameterised as:

        point(t) = centre + cos(t) * axis1 + sin(t) * axis2

    which handles both axis-aligned and rotated ellipses.
    """
    warnings.warn(
        f"Ellipse '{name or 'Un-named'}' will be approximated as a "
        f"PolygonOfInterest with {n_vertices} vertices.",
        UserWarning,
        stacklevel=3,
    )

    centre = (xy[0] + xy[2]) / 2
    axis1 = xy[0] - centre
    axis2 = xy[1] - centre

    angles = np.linspace(0, 2 * np.pi, n_vertices, endpoint=False)
    vertices = (
        centre
        + np.outer(np.cos(angles), axis1)
        + np.outer(np.sin(angles), axis2)
    )
    return PolygonOfInterest(vertices, name=name)
