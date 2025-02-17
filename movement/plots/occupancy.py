"""Wrappers for plotting occupancy data of select individuals."""

from collections.abc import Hashable, Sequence
from typing import Any, Literal, TypeAlias

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

HistInfoKeys: TypeAlias = Literal["counts", "xedges", "yedges"]

DEFAULT_HIST_ARGS = {"alpha": 1.0, "bins": 30, "cmap": "viridis"}


def plot_occupancy(
    da: xr.DataArray,
    individuals: Hashable | Sequence[Hashable] | None = None,
    keypoints: Hashable | Sequence[Hashable] | None = None,
    ax: plt.Axes | None = None,
    **kwargs: Any,
) -> tuple[plt.Figure, plt.Axes, dict[HistInfoKeys, np.ndarray]]:
    """Create a 2D histogram of the occupancy data given.

    By default;

    - If there are multiple keypoints selected, the occupancy of the centroid
        of these keypoints is computed.
    - If there are multiple individuals selected, the occupancies of their
        centroids are aggregated.

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

    For data with ``Nx`` bins in the 1st spatial dimension, and ``Ny`` bins in
    the 2nd spatial dimension, the dictionary output has key-value pairs;
    - ``xedges``, an ``(Nx+1,)`` ``numpy`` array specifying the bin edges in
    the 1st spatial dimension.
    - ``yedges``, an ``(Ny+1,)`` ``numpy`` array specifying the bin edges in
    the 2nd spatial dimension.
    - ``counts``, an ``(Nx, Ny)`` ``numpy`` array with the count for each bin.

    ``counts[x, y]`` is the number of datapoints in the
    ``(xedges[x], xedges[x+1]), (yedges[y], yedges[y+1])`` bin. These values
    are those returned from ``matplotlib.pyplot.Axes.hist2d``.

    See Also
    --------
    matplotlib.pyplot.Axes.hist2d : The underlying plotting function.

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

    # We need to remove NaN values from each individual, but we can't do this
    # right now because we still potentially have a (time, space, individuals)
    # array and so dropping NaNs along any axis may remove valid points for
    # other times / individuals.
    # Since we only care about a count, we can just unravel the individuals
    # dimension and create a "long" array of points. For example, a (10, 2, 5)
    # time-space-individuals DataArray becomes (50, 2).
    if "individuals" in data.dims:
        data = data.stack(
            {"new": ("time", "individuals")}, create_index=False
        ).rename({"new": "time"})
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
