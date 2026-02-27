"""Wrappers to plot movement data."""

import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
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
        as required dimensions. Optionally, it may have `individual` and/or
        `keypoint` dimensions.
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
    fig : matplotlib.figure.Figure or matplotlib.figure.SubFigure
        If ``ax`` is provided, this is ``ax.figure``
        (:class:`matplotlib.figure.Figure` or
        :class:`matplotlib.figure.SubFigure`). Otherwise, a new
        :class:`matplotlib.figure.Figure` is created and returned.
    ax : matplotlib.axes.Axes
        Axes on which the trajectory was drawn. If ``ax`` is provided,
        the input will be directly modified and returned in this value.

    """
    if isinstance(individual, list):
        raise ValueError("Only one individual can be selected.")

    selection = {}
    if "individual" in da.dims:
        selection["individual"] = individual or da.individual.values[0]
    if "keypoint" in da.dims:
        selection["keypoint"] = keypoints or da.keypoint.values

    plot_point = da.sel(selection)
    # If there are multiple selected keypoints, calculate the centroid
    plot_point = (
        plot_point.mean(dim="keypoint", skipna=True)
        if "keypoint" in plot_point.dims and plot_point.sizes["keypoint"] > 1
        else plot_point
    )
    plot_point = plot_point.squeeze()  # Only space and time should remain

    fig, ax = plt.subplots(figsize=(6, 6)) if ax is None else (ax.figure, ax)
    # Merge default plotting args with user-provided kwargs
    c_provided = "c" in kwargs
    kwargs = {**DEFAULT_PLOTTING_ARGS, **kwargs}
    kwargs.setdefault("c", plot_point.time.values[:, None])
    # Plot the scatter, colouring by time or user-provided colour
    sc = ax.scatter(
        plot_point.sel(space="x"),
        plot_point.sel(space="y"),
        **kwargs,
    )
    # Add 'colorbar' for time dimension if no colour was provided by user
    if not c_provided:
        cbar = fig.colorbar(sc, ax=ax, label="Time")
        if cbar.solids is not None:
            cbar.solids.set(alpha=1.0)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Trajectory")
    return fig, ax
