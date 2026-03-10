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
    from collections.abc import Sequence

    import napari.layers

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
        if present.
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
    to match the ellipse geometry. This approach was inspired by
    https://gis.stackexchange.com/questions/243459/drawing-ellipse-with-shapely

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


def shapes_layer_to_rois(
    layer: napari.layers.Shapes,
) -> list[LineOfInterest | PolygonOfInterest]:
    """Convert all shapes in a napari Shapes layer to movement RoI objects.

    Parameters
    ----------
    layer
        A napari Shapes layer, typically one created by ``RegionsWidget``
        (i.e. ``layer.metadata["movement_region_layer"] is True``).

    Returns
    -------
    list[LineOfInterest | PolygonOfInterest]
        One RoI per shape in the layer, in the same order as
        ``layer.data``.

    Notes
    -----
    Region names are read from ``layer.properties["name"]`` when present.
    Shapes whose name entry is missing or ``None`` receive the default
    name defined by :class:`~movement.roi.BaseRegionOfInterest`.

    See Also
    --------
    movement.roi.save_rois :
        Save the returned list directly to a GeoJSON file.
    rois_to_shapes_layer_data :
        Convert the returned list back to napari shapes (round-trip).

    """
    names = list(layer.properties.get("name", []))
    rois = []
    for i, (data, shape_type) in enumerate(
        zip(layer.data, layer.shape_type, strict=True)
    ):
        name = names[i] if i < len(names) else None
        rois.append(napari_shape_to_roi(data, shape_type, name=name))
    return rois


def roi_to_napari_shape(
    roi: BaseRegionOfInterest,
) -> tuple[np.ndarray, str]:
    """Convert a single movement RoI to a napari shape.

    Parameters
    ----------
    roi
        A :class:`~movement.roi.LineOfInterest` or
        :class:`~movement.roi.PolygonOfInterest`.

    Returns
    -------
    data : numpy.ndarray
        Shape coordinates in napari ``(y, x)`` convention.
    shape_type : str
        ``"path"`` for :class:`~movement.roi.LineOfInterest`,
        ``"polygon"`` for :class:`~movement.roi.PolygonOfInterest`.

    Notes
    -----
    The inverse of :func:`napari_shape_to_roi`.  Coordinates are swapped
    from movement ``(x, y)`` convention back to napari ``(y, x)``.

    """
    if isinstance(roi, PolygonOfInterest):
        # exterior.coords has a duplicate closing vertex; drop it
        xy = np.array(roi.coords)[:-1]
        shape_type = "polygon"
    else:
        xy = np.array(roi.coords)
        shape_type = "path"

    # Swap (x, y) → (y, x) to match napari's coordinate convention
    yx = xy[:, ::-1]
    return yx, shape_type


def rois_to_shapes_layer_data(
    rois: Sequence[BaseRegionOfInterest],
) -> dict:
    """Convert a sequence of movement RoIs to napari ``add_shapes`` kwargs.

    Parameters
    ----------
    rois
        Sequence of :class:`~movement.roi.LineOfInterest` and/or
        :class:`~movement.roi.PolygonOfInterest` objects.

    Returns
    -------
    dict
        A dictionary with keys ``"data"``, ``"shape_type"``, and
        ``"properties"`` (containing a ``"name"`` list).  It can be
        passed directly to ``viewer.add_shapes(**result)`` or used to
        populate an existing Shapes layer.

    See Also
    --------
    movement.roi.load_rois :
        Load RoIs from a GeoJSON file; the result can be passed directly
        to this function to populate a shapes layer.
    shapes_layer_to_rois :
        Convert a shapes layer to RoIs (inverse direction).

    """
    data = []
    shape_types = []
    names = []
    for roi in rois:
        yx, shape_type = roi_to_napari_shape(roi)
        data.append(yx)
        shape_types.append(shape_type)
        names.append(roi.name)
    return {
        "data": data,
        "shape_type": shape_types,
        "properties": {"name": names},
    }
