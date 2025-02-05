"""Wrappers to plot movement data."""

from pathlib import Path

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
    selection: dict[str, str | int | list[str | int] | None]
    | None = None,  # TODO: make less complex!
    image_path: None | Path = None,
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
        (e.g., "individuals", "keypoints") and the values should be the index
        or name of an individual and a keypoint, or a list of keypoints
        (their centroid will be plotted). By default, the first individual is
        chosen and the centroid of all keypoints is plotted. If there is no
        `individuals` or `keypoints` dimension, this argument is ignored.
    image_path : None or Path, optional
        Path to an image over which the trajectory data can be overlaid,
        e.g., a reference video frame.
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
    selection.setdefault(
        "individuals",
        str(da.individuals.values[0]) if "individuals" in da.dims else None,
    )  # First individual by index
    selection.setdefault("keypoints", None) if "keypoints" in da.dims else None

    chosen_ind = selection["individuals"] if "individuals" in da.dims else None
    title_suffix = f" of {chosen_ind}" if chosen_ind is not None else ""

    # Determine which keypoint(s) to select (if any)
    selection["keypoints"] = (
        (
            da.keypoints.values
            if selection["keypoints"] is None
            else [selection["keypoints"]]
            if isinstance(selection["keypoints"], str)
            else selection["keypoints"]
        )
        if "keypoints" in da.dims
        else None
    )

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
    kwargs = {
        key: kwargs.setdefault(key, value)
        for key, value in DEFAULT_PLOTTING_ARGS.items()
    }

    # Plot the scatter, coloring by time
    sc = ax.scatter(
        plot_point.sel(space="x"),
        plot_point.sel(space="y"),
        c=plot_point.time,
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
    colorbar = fig.colorbar(sc, ax=ax, label=time_label)
    # Ensure colorbar is fully opaque
    colorbar.solids.set(alpha=1.0)

    if image_path is not None:
        frame = plt.imread(image_path)
        # Invert the y-axis back again since the the image is plotted
        # using a coordinate system with origin on the top left of the image
        ax.invert_yaxis()
        ax.imshow(frame)

    return fig, ax
