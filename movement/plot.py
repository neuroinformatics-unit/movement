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
    keypoint: str | list[str],
    individual: int | str = 0,
    frame_path: None | Path = None,
    title: str | None = None,
    **kwargs,
) -> plt.Figure:
    """Plot trajectory.

    This function plots the  trajectory of a specified keypoint or the midpoint
    between two keypoints for a given individual. The individual can be
    specified by their index or name. By default, the first individual
    is selected.

    Parameters
    ----------
    da: xr.DataArray
        Movement poses data.
    keypoint: str | list[str]
        Either one or two keypoints, from which to form the trajectory.
        If a single keypoint is given, a trajectory of that keypoint is
        plotted. If two keypoints are given, the trajectory of their centroid
        is plotted.
    individual: int | str
        Individual index or name. By default, the first individual is chosen.
    frame_path: None | Path, optional
        Path to the frame image for the trajectory data to be overlaid on top
        of.
    title: str | None, optional
        Title of the plot. If no title is provided, it is generated based on
        the keypoint names and individual name.
    kwargs: Any
        Arguments passed to ``matplotlib.pyplot.Figure`` when creating the
        plot.

    Returns
    -------
    matplotlib.pyplot.Figure
        Figure handler for the created plot.

    """
    position = da.position
    if isinstance(keypoint, list):
        plotting_point = position.sel(keypoints=keypoint).mean(dim="keypoints")
        plotting_point_name = "midpoint between " + " and ".join(keypoint)
    elif isinstance(keypoint, str):
        plotting_point = position.sel(keypoints=keypoint)
        plotting_point_name = keypoint

    # recover individual's coordinate (name), if provided an integer
    individual = (
        da.individuals.values[individual]
        if isinstance(individual, int)
        else str(individual)
    )

    fig, ax = plt.subplots(1, 1)

    # if a key us not in kwargs use default plotting arg for that key
    for key, value in DEFAULT_PLOTTING_ARGS.items():
        if key not in kwargs:
            kwargs[key] = value

    sc = ax.scatter(
        plotting_point.sel(individuals=individual, space="x"),
        plotting_point.sel(individuals=individual, space="y"),
        c=plotting_point.time,
        **kwargs,
    )

    ax.axis("equal")
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.invert_yaxis()
    title = (
        f"{individual} trajectory of {plotting_point_name}"
        if title is not None
        else title
    )
    ax.set_title(title)

    label = (
        f"time ({da.attrs['time_unit']})"
        if da.attrs.get("time_unit") is not None
        else "time steps (frames)"
    )
    colourbar = fig.colorbar(sc, ax=ax, label=label)
    # ensure colourbar does not suffer from transparency
    colourbar.solids.set(alpha=1.0)

    if frame_path is not None:
        frame = plt.imread(frame_path)
        # Invert the y-axis back again since the the image is plotted
        # using a coordinate system with origin on the top left of the image
        ax.invert_yaxis()
        ax.imshow(frame)

    return fig
