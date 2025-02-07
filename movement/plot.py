"""Wrappers to plot movement data."""

import xarray as xr
from matplotlib import pyplot as plt

DEFAULT_PLOTTING_ARGS = {
    "s": 15,
    "cmap": "viridis",
    "marker": "o",
    "alpha": 1.0,
}


def trajectory(
    da: xr.DataArray,
    selection: dict[str, str | list[str] | None]
    | None = None,  # TODO: make less complex!
    ax: plt.Axes | None = None,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot trajectory.

    This function plots the trajectory of a specified keypoint or the centroid
    between multiple keypoints for a given individual. The individual can be
    specified by their index or name. By default, the first individual is
    selected. The trajectory is colored by time (using the default colormap).
    Pass a different colormap through ``cmap`` if desired.

    Parameters
    ----------
    da : xr.DataArray
        A data array containing position information, with `time` and `space`
        as required dimensions. Optionally, it may have `individuals` and/or
        `keypoints` dimensions.
    selection : dict, optional
        A dictionary specifying the selection criteria for the individual and
        point to plot. The dictionary keys should be the dimension names
        (e.g., "individuals", "keypoints") and the values should be the
        individual or keypoint names (as strings), or a list of keypoint names
        (their centroid will be plotted). By default, the first individual is
        chosen and the centroid of all keypoints is plotted. If there is no
        `individuals` or `keypoints` dimension, this argument is ignored.
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
    # Construct selection dict for individuals and keypoints
    selection = selection or {}

    # Set default values for individuals and keypoints if they are in da.dims
    if "individuals" in da.dims:
        selection.setdefault(
            "individuals", str(da.individuals.values[0])
        )  # First individual by index

    title_suffix = (
        f" of {selection['individuals']}" if "individuals" in da.dims else ""
    )

    # Determine which keypoint(s) to select (if any)
    if "keypoints" in da.dims:
        if "keypoints" not in selection or selection["keypoints"] is None:
            selection["keypoints"] = da.keypoints.values
        elif isinstance(selection["keypoints"], str):
            selection["keypoints"] = [selection["keypoints"]]

    # Select the data for the specified individual and keypoint(s)
    plot_point = da.sel(**selection)

    # If there are multiple selected keypoints, calculate the centroid
    plot_point = (
        plot_point.mean(dim="keypoints", skipna=True)
        if "keypoints" in plot_point.dims and plot_point.sizes["keypoints"] > 1
        else plot_point
    )

    # Squeeze all dimensions with size 1 (only time and space should remain)
    plot_point = plot_point.squeeze()

    # Create a new Figure/Axes if none is passed
    fig, ax = plt.subplots(figsize=(6, 6)) if ax is None else (ax.figure, ax)

    # Merge default plotting args with user-provided kwargs
    for key, value in DEFAULT_PLOTTING_ARGS.items():
        kwargs.setdefault(key, value)

    colorbar = False
    if "c" not in kwargs:
        kwargs["c"] = plot_point.time
        colorbar = True

    # Plot the scatter, coloring by time or user-provided color
    sc = ax.scatter(
        plot_point.sel(space="x"),
        plot_point.sel(space="y"),
        **kwargs,
    )

    # Handle axis labeling
    space_unit = da.attrs.get("space_unit", "pixels")
    ax.set_xlabel(f"x ({space_unit})")
    ax.set_ylabel(f"y ({space_unit})")
    ax.axis("equal")

    # By default, invert y-axis so (0,0) is in the top-left,
    # matching typical image coordinate systems
    ax.invert_yaxis()

    # Generate default title if none provided
    ax.set_title(f"Trajectory{title_suffix}")

    # Add colorbar for time dimension
    time_unit = da.attrs.get("time_unit")
    time_label = f"time ({time_unit})" if time_unit else "time steps (frames)"
    fig.colorbar(sc, ax=ax, label=time_label).solids.set(
        alpha=1.0
    ) if colorbar else None

    return fig, ax
