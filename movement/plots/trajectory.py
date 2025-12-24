"""Wrappers to plot movement data."""

from typing import cast

import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure, SubFigure

DEFAULT_PLOTTING_ARGS = {
    "s": 15,
    "marker": "o",
    "alpha": 1.0,
}


def plot_centroid_trajectory(
    da: xr.DataArray,
    individual: str | None = None,
    keypoints: str | list[str] | None = None,
    ax: Axes | None = None,
    **kwargs,
) -> tuple[Figure | SubFigure, Axes]:
    """Plot centroid trajectory.

    This function plots the trajectory of the centroid
    of multiple keypoints for a given individual. By default, the trajectory
    is colored by time (using the default colormap). Pass a different colormap
    through ``cmap`` if desired. If a single keypoint is passed, the trajectory
    will be the same as the trajectory of the keypoint.

    Parameters
    ----------
    da : xr.DataArray
        A data array containing position information, with `time` and `space`
        as required dimensions. Optionally, it may have `individuals` and/or
        `keypoints` dimensions.
    individual : str, optional
        The name of the individual to be plotted. By default, the first
        individual is plotted.
    keypoints : str, list[str], optional
        The name of the keypoint to be plotted, or a list of keypoint names
        (their centroid will be plotted). By default, the centroid of all
        keypoints is plotted.
    ax : matplotlib.axes.Axes or None, optional
        Axes object on which to draw the trajectory. If None, a new
        figure and axes are created.
    **kwargs : dict
        Additional keyword arguments passed to
        :meth:`matplotlib.axes.Axes.scatter`.

    Returns
    -------
    (figure, axes) : tuple of (matplotlib.figure.Figure | SubFigure, matplotlib.axes.Axes)
        The figure and axes containing the trajectory plot.

    """
    if isinstance(individual, list):
        raise ValueError("Only one individual can be selected.")

    selection = {}

    if "individuals" in da.dims:
        if individual is None:
            selection["individuals"] = da.individuals.values[0]
        else:
            selection["individuals"] = individual

    if "keypoints" in da.dims:
        if keypoints is None:
            selection["keypoints"] = da.keypoints.values
        else:
            selection["keypoints"] = keypoints

    plot_point = da.sel(**selection)

    # If there are multiple selected keypoints, calculate the centroid
    plot_point = (
        plot_point.mean(dim="keypoints", skipna=True)
        if "keypoints" in plot_point.dims and plot_point.sizes["keypoints"] > 1
        else plot_point
    )

    plot_point = plot_point.squeeze()  # Only space and time should remain

    fig, ax = plt.subplots(figsize=(6, 6)) if ax is None else (ax.figure, ax)

    # Merge default plotting args with user-provided kwargs
    for key, value in DEFAULT_PLOTTING_ARGS.items():
        kwargs.setdefault(key, value)

    colorbar = False
    if "c" not in kwargs:
        # set color by time if not provided
        kwargs["c"] = plot_point.time
        colorbar = True

    # Plot the scatter, colouring by time or user-provided colour
    sc: PathCollection = ax.scatter(
        plot_point.sel(space="x"),
        plot_point.sel(space="y"),
        **kwargs,
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Trajectory")

    # Add 'colorbar' for time dimension if no colour was provided by user
    time_label = "Time"
    if colorbar:
        cb: Colorbar | None = fig.colorbar(sc, ax=ax, label=time_label)
        if cb is not None and hasattr(cb, "solids") and cb.solids is not None:
            try:
                cb.solids.set(alpha=1.0)
            except Exception:
                # some backends or colorbars might not support solids.set(); ignore safely
                pass

    return cast(Figure | SubFigure, fig), ax
