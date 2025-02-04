"""Wrappers to plot movement data."""

from typing import Any, Literal, TypeAlias

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

HistInfoKeys: TypeAlias = Literal["counts", "xedges", "yedges"]

DEFAULT_HIST_ARGS = {"alpha": 1.0, "bins": 30, "cmap": "viridis"}


def occupancy_histogram(
    da: xr.DataArray,
    keypoint: int | str = 0,
    individual: int | str = 0,
    title: str | None = None,
    ax: plt.Axes | None = None,
    **kwargs: Any,
) -> tuple[plt.Figure, plt.Axes, dict[HistInfoKeys, np.ndarray]]:
    """Create a 2D histogram of the occupancy data given.

    Time-points whose corresponding spatial coordinates have NaN values
    are ignored. Histogram information is returned as the second output
    value (see Notes).

    Parameters
    ----------
    da : xarray.DataArray
        Spatial data to create histogram for. NaN values are dropped.
    keypoint : int | str
        The keypoint to create a histogram for.
    individual : int | str
        The individual to create a histogram for.
    title : str, optional
        Title to give to the plot. Default will be generated from the
        ``keypoint`` and ``individual``
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
    data = da.position if isinstance(da, xr.Dataset) else da
    title_components = []

    # Remove additional dimensions before dropping NaN values
    if "individuals" in data.dims:
        if individual not in data["individuals"]:
            individual = data["individuals"].values[individual]
        data = data.sel(individuals=individual).squeeze()
        title_components.append(f"individual {individual}")
    if "keypoints" in data.dims:
        if keypoint not in data["keypoints"]:
            keypoint = data["keypoints"].values[keypoint]
        data = data.sel(keypoints=keypoint).squeeze()
        title_components.append(f"keypoint {keypoint}")
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

    # Axis labels and title
    if not title and title_components:
        title = "Occupancy of " + ", ".join(title_components)
    if title:
        ax.set_title(title)
    space_unit = data.attrs.get("space_unit", "pixels")
    ax.set_xlabel(f"{x_coord} ({space_unit})")
    ax.set_ylabel(f"{y_coord} ({space_unit})")

    return fig, ax, {"counts": counts, "xedges": xedges, "yedges": yedges}
