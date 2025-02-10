"""Wrappers to plot movement data."""

import xarray as xr
from matplotlib import pyplot as plt

DEFAULT_PLOTTING_ARGS = {
    "s": 15,
    "cmap": "viridis",
    "marker": "o",
    "alpha": 1.0,
}


def plot_trajectory(
    da: xr.DataArray,
    keypoints: str | list[str] | None = None,
    individual: str | None = None,
    ax: plt.Axes | None = None,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot trajectory.

    This function plots the trajectory of a specified keypoint or the centroid
    between multiple keypoints for a given individual. By default, the first
    individual, and the centroid of all keypoints is selected. The trajectory
    is colored by time (using the default colormap).Pass a different colormap
    through ``cmap`` if desired.

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
        ``matplotlib.axes.Axes.scatter()``.

    Returns
    -------
    (figure, axes) : tuple of (matplotlib.pyplot.Figure, matplotlib.axes.Axes)
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

    title_suffix = f" of {individual}" if "individuals" in da.dims else ""

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
        kwargs["c"] = plot_point.time
        colorbar = True

    # Plot the scatter, colouring by time or user-provided colour
    sc = ax.scatter(
        plot_point.sel(space="x"),
        plot_point.sel(space="y"),
        **kwargs,
    )

    space_unit = da.attrs.get("space_unit", "pixels")
    ax.set_xlabel(f"x ({space_unit})")
    ax.set_ylabel(f"y ({space_unit})")
    ax.axis("equal")
    ax.set_title(f"Trajectory{title_suffix}")

    # Add 'colorbar' for time dimension if no colour was provided by user
    time_unit = da.attrs.get("time_unit")
    time_label = f"time ({time_unit})" if time_unit else "time steps (frames)"
    fig.colorbar(sc, ax=ax, label=time_label).solids.set(
        alpha=1.0
    ) if colorbar else None

    return fig, ax
