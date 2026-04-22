"""Conversion functions between ``napari`` shapes and ``movement`` RoIs.

- For information on ``napari`` shapes,
  see https://napari.org/stable/howtos/layers/shapes.html
- RoI: Region of Interest, as defined in :mod:`movement.roi`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import shapely
import shapely.affinity

from movement.roi.line import LineOfInterest
from movement.roi.polygon import PolygonOfInterest
from movement.utils.logging import logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    from napari.layers import Shapes

    from movement.roi.base import BaseRegionOfInterest

type NapariShapeType = Literal[
    "line", "path", "polygon", "rectangle", "ellipse"
]

NAPARI_SHAPE_TO_ROI_CLASS: dict[
    NapariShapeType, type[BaseRegionOfInterest]
] = {
    "line": LineOfInterest,
    "path": LineOfInterest,
    "polygon": PolygonOfInterest,
    "rectangle": PolygonOfInterest,
    "ellipse": PolygonOfInterest,  # approximated as polygon
}


def roi_to_napari_shape(
    roi: BaseRegionOfInterest,
) -> tuple[np.ndarray, NapariShapeType]:
    """Convert a ``movement`` RegionOfInterest (RoI) to a ``napari`` shape.

    Parameters
    ----------
    roi
        The region of interest to convert, e.g.
        :class:`~movement.roi.LineOfInterest`,
        :class:`~movement.roi.PolygonOfInterest`.

    Returns
    -------
    data : numpy.ndarray
        Shape coordinates as an (N, 2) array in ``(y, x)`` order
        (``napari`` convention), with no repeated closing vertex.
    shape_type : NapariShapeType
        The ``napari`` shape type string: ``"path"`` for
        :class:`~movement.roi.LineOfInterest` and ``"polygon"`` for
        :class:`~movement.roi.PolygonOfInterest`.

    Notes
    -----
    The mapping from ``movement`` RoI classes to ``napari`` shape types is:

    .. list-table::
       :header-rows: 1

       * - movement RoI class
         - napari shape type
       * - :class:`~movement.roi.LineOfInterest`
         - ``"path"``
       * - :class:`~movement.roi.PolygonOfInterest`
         - ``"polygon"``

    This function is the inverse of :func:`napari_shape_to_roi`, but some
    shape information is not preserved when converting back. Specifically,
    ``"line"``, ``"rectangle"``, and ``"ellipse"`` shapes drawn in ``napari``
    are all returned as ``"path"`` or ``"polygon"``.

    A closed :class:`~movement.roi.LineOfInterest` (created with
    ``loop=True``) is also affected: ``napari`` has no closed-path shape type,
    so the segment connecting the last point back to the first is dropped
    and a warning is emitted.

    See Also
    --------
    napari_shape_to_roi : The inverse of this function.
    rois_to_napari_shapes : Batch conversion of multiple RoIs.

    """
    xy = np.array(roi.coords)

    if isinstance(roi, PolygonOfInterest):
        shape_type: NapariShapeType = "polygon"
        xy = xy[:-1]  # shapely Polygon exterior repeats the first vertex
    else:
        shape_type = "path"
        if roi.is_closed:
            xy = xy[:-1]  # shapely LinearRing repeats the first vertex
            logger.warning(
                f"LineOfInterest '{roi.name}' is a closed loop, but napari "
                f"has no closed-path shape type. Converting to 'path'; the "
                f"closing segment will not be shown in napari."
            )

    # Swap (x, y) → (y, x) to match napari's coordinate convention
    return xy[:, ::-1], shape_type


def rois_to_napari_shapes(
    rois: Sequence[BaseRegionOfInterest],
) -> dict[str, Any]:
    """Convert a sequence of ``movement`` RoIs to ``napari`` shapes.

    The returned dictionary can be passed directly to ``napari``'s Shapes layer
    constructor to add all regions of interest (RoIs) in a single call.

    Parameters
    ----------
    rois
        Sequence of :class:`~movement.roi.LineOfInterest` or
        :class:`~movement.roi.PolygonOfInterest` objects to convert.

    Returns
    -------
    dict
        A dictionary with the following keys:

        - ``"data"``: list of (N, 2) arrays in ``(y, x)`` order.
        - ``"shape_type"``: list of ``napari`` shape type strings.
        - ``"properties"``: dict with a ``"name"`` key containing the
          RoI names.

    See Also
    --------
    roi_to_napari_shape : The underlying per-shape conversion function.
    napari_shapes_to_rois : The inverse of this function.

    """
    data, shape_types, names = [], [], []
    for roi in rois:
        coords, shape_type = roi_to_napari_shape(roi)
        data.append(coords)
        shape_types.append(shape_type)
        names.append(roi.name)
    return {
        "data": data,
        "shape_type": shape_types,
        "properties": {"name": names},
    }


def napari_shape_to_roi(
    data: np.ndarray,
    shape_type: NapariShapeType,
    name: str | None = None,
) -> BaseRegionOfInterest:
    """Convert a ``napari`` shape to a ``movement`` RegionOfInterest (RoI).

    This function only handles static 2D shapes with coordinates (y, x).
    Shapes with additional dimensions will be rejected with an error.

    Parameters
    ----------
    data
        Shape coordinates as stored in ``layer.data[i]``, where ``i`` is the
        index of the shape to convert. This should be an (N, 2) array where
        rows are vertices and columns are (y, x) coordinates.
    shape_type
        One of the ``napari`` shape types.
    name
        Name to assign to the resulting RoI. If ``None``, the RoI
        receives the default name defined by
        :class:`~movement.roi.BaseRegionOfInterest`.

    Returns
    -------
    BaseRegionOfInterest
        An instance of a :class:`~movement.roi.BaseRegionOfInterest`
        subclass corresponding to the input ``shape_type``, e.g.
        :class:`~movement.roi.LineOfInterest`,
        :class:`~movement.roi.PolygonOfInterest`.

    Raises
    ------
    ValueError
        If ``data`` has more than 2 columns (dimensions other than y and x),
        or if ``shape_type`` is not one of the recognised ``napari`` shape
        types.

    Notes
    -----
    The mapping from ``napari`` shape types to ``movement`` RoI classes is:

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
    The approximation uses :meth:`shapely.Point.buffer` scaled and rotated
    to match the ellipse geometry. This approach was inspired by
    https://gis.stackexchange.com/questions/243459/drawing-ellipse-with-shapely

    See Also
    --------
    roi_to_napari_shape : The inverse of this function.
    napari_shapes_to_rois : Batch conversion of an entire ``napari`` layer.

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

    return roi_class(xy, name=name or None)


def napari_shapes_to_rois(
    layer: Shapes,
) -> list[BaseRegionOfInterest]:
    """Convert all shapes in a ``napari`` Shapes layer to ``movement`` RoIs.

    Parameters
    ----------
    layer
        The ``napari`` Shapes layer to be converted. Names are read from
        ``layer.properties["name"]`` when available. Missing or blank names
        receive the default name defined by
        :class:`~movement.roi.BaseRegionOfInterest`.

    Returns
    -------
    list[BaseRegionOfInterest]
        One region of interest (RoI) per shape in the layer, in the same order.

    Raises
    ------
    ValueError
        If any shape has more than 2 coordinate columns, or has an
        unrecognised shape type. See :func:`napari_shape_to_roi`.

    See Also
    --------
    napari_shape_to_roi : The underlying per-shape conversion function.
    rois_to_napari_shapes : The inverse of this function.

    """
    names = list(layer.properties.get("name", []))
    return [
        napari_shape_to_roi(
            data,
            shape_type,
            name=names[i] if i < len(names) else None,
        )
        for i, (data, shape_type) in enumerate(
            zip(layer.data, layer.shape_type, strict=True)
        )
    ]


def _ellipse_to_polygon(xy: np.ndarray) -> np.ndarray:
    """Approximate a ``napari`` ellipse as a polygon, returning the vertices.

    ``napari`` stores an ellipse as the 4 corners of its bounding rectangle.
    After the (y, x) → (x, y) swap has already been applied, these are
    four corners such that ``xy[0]`` and ``xy[2]`` are diagonally opposite.

    The semi-axis lengths are derived from the side lengths of the bounding
    rectangle (the ellipse is inscribed in it). A unit circle is created at
    the centre, scaled to the semi-axis lengths, and rotated to match the
    rectangle orientation. The resulting polygon's vertices are returned as
    an (N, 2) array of (x, y), without repeating the first vertex at the
    end.
    """
    centre = (xy[0] + xy[2]) / 2
    side_a = xy[1] - xy[0]  # one edge of the bounding rectangle
    side_b = xy[3] - xy[0]  # adjacent edge
    semi_a = float(np.linalg.norm(side_a)) / 2
    semi_b = float(np.linalg.norm(side_b)) / 2
    angle = float(np.degrees(np.arctan2(side_a[1], side_a[0])))

    ellipse_polygon = shapely.affinity.rotate(
        shapely.affinity.scale(
            shapely.Point(centre).buffer(1),
            semi_a,
            semi_b,
        ),
        angle,
    )

    return np.array(ellipse_polygon.exterior.coords)[:-1]
