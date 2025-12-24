"""Wrappers for plotting occupancy data of select individuals."""

from collections.abc import Hashable, Sequence
from contextlib import suppress
from typing import Any, Literal, TypeAlias, cast

import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure, SubFigure

HistInfoKeys: TypeAlias = Literal["counts", "xedges", "yedges"]

DEFAULT_HIST_ARGS = {"alpha": 1.0, "bins": 30, "cmap": "viridis"}


def plot_occupancy(  # noqa: C901
    da: xr.DataArray,
    individuals: Hashable | Sequence[Hashable] | None = None,
    keypoints: Hashable | Sequence[Hashable] | None = None,
    ax: Axes | None = None,
    **kwargs: Any,
) -> tuple[Figure | SubFigure, Axes, dict[HistInfoKeys, np.ndarray]]:
    """Create a 2D occupancy histogram.

    - If there are multiple keypoints selected, the occupancy of the centroid
      of these keypoints is computed.
    - If there are multiple individuals selected, their occupancies are
      aggregated.

    Points whose corresponding spatial coordinates have NaN values
    are ignored.

    Histogram information is returned as the third output value (see Notes).

    Parameters
    ----------
    da : xarray.DataArray
        Spatial data to create histogram for. NaN values are dropped.
    individuals : Hashable, optional
        The name of the individual(s) to be aggregated and plotted. By default,
        all individuals are aggregated.
    keypoints : Hashable | list[Hashable], optional
        Name of a keypoint or list of such names. The centroid of all provided
        keypoints is computed, then plotted in the histogram.
    ax : matplotlib.axes.Axes, optional
        Axes object on which to draw the histogram. If not provided, a new
        figure and axes are created and returned.
    kwargs : Any
        Keyword arguments passed to :meth:`matplotlib.axes.Axes.hist2d`.

    Returns
    -------
    matplotlib.pyplot.Figure | SubFigure, matplotlib.axes.Axes, dict[HistInfoKeys, np.ndarray]
        The figure and axes and histogram info (counts, xedges, yedges).

    """
    # Collapse dimensions if necessary
    data = da.copy(deep=True)
    if "keypoints" in da.dims:
        if keypoints is not None:
            data = data.sel(keypoints=keypoints)
        # A selection of just one keypoint automatically drops the keypoints
        # dimension, hence the need to re-check this here
        if "keypoints" in data.dims:
            data = data.mean(dim="keypoints", skipna=True)
    if "individuals" in da.dims and individuals is not None:
        data = data.sel(individuals=individuals)

    # Unravel individuals into a long time axis if present
    if "individuals" in data.dims:
        data = data.stack(
            {"new": ("time", "individuals")}, create_index=False
        ).swap_dims({"new": "time"})
    data = data.dropna(dim="time", how="any")

    # This makes us agnostic to the planar coordinate system.
    x_coord = data["space"].values[0]
    y_coord = data["space"].values[1]

    # Inherit our defaults if not otherwise provided
    for key, value in DEFAULT_HIST_ARGS.items():
        if key not in kwargs:
            kwargs[key] = value

    # Create figure/axes if necessary
    if ax is not None:
        fig = ax.get_figure()
        if fig is None:
            fig, ax = plt.subplots()
    else:
        fig, ax = plt.subplots()

    # Create histogram
    counts, xedges, yedges, hist_image = ax.hist2d(
        data.sel(space=x_coord), data.sel(space=y_coord), **kwargs
    )

    # hist_image may be a QuadMesh or similar mappable; guard optional access
    # colorbar() may return a Colorbar or None depending on backend
    cb: Colorbar | None = None
    try:
        cb = fig.colorbar(hist_image, ax=ax)
    except Exception:
        # Some backends might fail to create a colorbar for this mappable;
        # ignore this safely.
        cb = None

    if cb is not None and hasattr(cb, "solids") and cb.solids is not None:
        with suppress(Exception):
            cb.solids.set(alpha=1.0)

    # Some mappables may support set(); guard before calling
    if hist_image is not None and hasattr(hist_image, "set"):
        with suppress(Exception):
            # cast to QuadMesh for type-checkers when appropriate
            _img = cast(QuadMesh, hist_image)
            _img.set(alpha=kwargs.get("alpha", 1.0))

    ax.set_xlabel(str(x_coord))
    ax.set_ylabel(str(y_coord))

    return (
        cast(Figure | SubFigure, fig),
        ax,
        {"counts": counts, "xedges": xedges, "yedges": yedges},
    )
