"""Wrappers for plotting occupancy data of select individuals."""

from collections.abc import Hashable
from typing import Any, Literal, TypeAlias

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

HistInfoKeys: TypeAlias = Literal["counts", "xedges", "yedges"]

DEFAULT_HIST_ARGS = {"alpha": 1.0, "bins": 30, "cmap": "viridis"}


def plot_occupancy(
    da: xr.DataArray,
    selection: dict[str, Hashable] | None = None,
    ax: plt.Axes | None = None,
    **kwargs: Any,
) -> tuple[plt.Figure, plt.Axes, dict[HistInfoKeys, np.ndarray]]:
    """Create a 2D histogram of the occupancy data given.

    By default, the 0-indexed value along non-"time" and non-"space" dimensions
    is plotted. The ``selection`` variable can be used to select different
    coordinates along additional dimensions to plot instead.

    Time-points whose corresponding spatial coordinates have NaN values
    are ignored.

    Histogram information is returned as the third output value (see Notes).

    Parameters
    ----------
    da : xarray.DataArray
        Spatial data to create histogram for. NaN values are dropped.
    selection : dict[str, Hashable], optional
        Mapping of dimension identifiers to the coordinate along that dimension
        to plot. "time" and "space" dimensions are ignored. For example,
        ``selection = {"individuals": "Bravo"}`` will create the occupancy
        histogram for the individual "Bravo", instead of the occupancy
        histogram for the 0-indexed entry on the ``"individuals"`` dimension.
    ax : matplotlib.axes.Axes, optional
        Axes object on which to draw the histogram. If not provided, a new
        figure and axes are created and returned.
    kwargs : Any
        Keyword arguments passed to ``matplotlib.pyplot.hist2d``

    Returns
    -------
    matplotlib.pyplot.Figure
        Plot handle containing the rendered 2D histogram. If ``ax`` is
        supplied, this will be the figure that ``ax`` belongs to.
    matplotlib.axes.Axes
        Axes on which the histogram was drawn. If ``ax`` was supplied,
        the input will be directly modified and returned in this value.
    dict[str, numpy.ndarray]
        Information about the created histogram (see Notes).

    Notes
    -----
    In instances where the counts or information about the histogram bins is
    desired, the ``return_hist_info`` argument should be provided as ``True``.
    This will force the function to return a second output value, which is a
    dictionary containing the bin edges and bin counts that were used to create
    the histogram.

    For data with ``N`` time-points, the dictionary output has key-value pairs;
    - ``xedges``, an ``(N+1,)`` ``numpy`` array specifying the bin edges in the
    first spatial dimension.
    - ``yedges``, same as ``xedges`` but for the second spatial dimension.
    - ``counts``, an ``(N, N)`` ``numpy`` array with the count for each bin.

    ``counts[x, y]`` is the number of datapoints in the
    ``(xedges[x], xedges[x+1]), (yedges[y], yedges[y+1])`` bin. These values
    are those returned from ``matplotlib.pyplot.Axes.hist2d``.

    Note that the ``counts`` values do not necessarily match the mappable
    values that one gets from extracting the data from the
    ``matplotlib.collections.QuadMesh`` object (that represents the rendered
    histogram) via its ``get_array()`` attribute.

    See Also
    --------
    matplotlib.pyplot.Axes.hist2d : The underlying plotting function.

    """
    if selection is None:
        selection = dict()

    # Remove additional dimensions before dropping NaN values
    non_spacetime_dims = [
        dim for dim in da.dims if dim not in ("time", "space")
    ]
    selection = {
        dim: selection.get(dim, da[dim].values[0])
        for dim in non_spacetime_dims
    }
    data: xr.DataArray = da.sel(**selection).squeeze()
    # Selections must be scalar, resulting in 2D data.
    # Catch this now
    if data.ndim != 2:
        raise IndexError(
            "Histogram data was not time-space only. "
            "Did you accidentally pass multiple coordinates for any of "
            f"the following dimensions: {non_spacetime_dims}"
        )

    # We should now have just the relevant time-space data,
    # so we can remove time-points with NaN values.
    data = data.dropna(dim="time", how="any")
    # This makes us agnostic to the planar coordinate system.
    x_coord = data["space"].values[0]
    y_coord = data["space"].values[1]

    # Inherit our defaults if not otherwise provided
    for key, value in DEFAULT_HIST_ARGS.items():
        if key not in kwargs:
            kwargs[key] = value
    # Now it should just be a case of creating the histogram
    if ax is not None:
        fig = ax.get_figure()
    else:
        fig, ax = plt.subplots()
    counts, xedges, yedges, hist_image = ax.hist2d(
        data.sel(space=x_coord), data.sel(space=y_coord), **kwargs
    )
    colourbar = fig.colorbar(hist_image, ax=ax)
    colourbar.solids.set(alpha=1.0)

    space_unit = data.attrs.get("space_unit", "pixels")
    ax.set_xlabel(f"{x_coord} ({space_unit})")
    ax.set_ylabel(f"{y_coord} ({space_unit})")

    return fig, ax, {"counts": counts, "xedges": xedges, "yedges": yedges}
