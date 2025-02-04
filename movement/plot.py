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
    individual: int | str = 0,
    keypoint: None | str | list[str] = None,
    image_path: None | Path = None,
    title: str | None = None,
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
    individual : int or str, default=0
        Individual index or name. By default, the first individual is chosen.
        If there is no `individuals` dimension, this argument is ignored.
    keypoint : None, str or list of str, optional
        - If None, the centroid of **all** keypoints is plotted (default).
        - If str, that single keypoint's trajectory is plotted.
        - If a list of keypoints, their centroid is plotted.
        If there is no `keypoints` dimension, this argument is ignored.
    image_path : None or Path, optional
        Path to an image over which the trajectory data can be overlaid,
        e.g., a reference video frame.
    title : str or None, optional
        Title of the plot. If not provided, one is generated based on
        the individual's name (if applicable).
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
    selection = {}

    # Determine which individual to select (if any)
    if "individuals" in da.dims:
        # Convert int index to actual individual name if needed
        selection["individuals"] = chosen_ind = (
            da.individuals.values[individual]
            if isinstance(individual, int)
            else str(individual)
        )
        title_suffix = f" of {chosen_ind}"
    else:
        title_suffix = ""

    # Determine which keypoint(s) to select (if any)
    if "keypoints" in da.dims:
        if keypoint is None:
            selection["keypoints"] = da.keypoints.values
        elif isinstance(keypoint, str):
            selection["keypoints"] = [keypoint]
        elif isinstance(keypoint, list):
            selection["keypoints"] = keypoint

    # Select the data for the specified individual and keypoint(s)
    plot_point = da.sel(**selection)

    # If there are multiple selected keypoints, calculate the centroid
    if "keypoints" in plot_point.dims and plot_point.sizes["keypoints"] > 1:
        plot_point = plot_point.mean(dim="keypoints", skipna=True)

    # Squeeze all dimensions with size 1 (only time and space should remain)
    plot_point = plot_point.squeeze()

    # Create a new Figure/Axes if none is passed
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

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
    if title is None:
        title = f"Trajectory{title_suffix}"
    ax.set_title(title)

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
