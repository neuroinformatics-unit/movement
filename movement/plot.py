"""Wrappers to plot movement data."""

from typing import Any

import matplotlib.pyplot as plt
import xarray as xr

DEFAULT_HIST_ARGS = {"alpha": 1.0, "bins": 30, "cmap": "viridis"}


def occupancy_histogram(
    da: xr.DataArray,
    keypoint: int | str = 0,
    individual: int | str = 0,
    title: str | None = None,
    **kwargs: Any,
) -> plt.Figure:
    """Create a 2D histogram of the occupancy data given.

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
    kwargs : Any
        Keyword arguments passed to ``matplotlib.pyplot.hist2d``

    Returns
    -------
    matplotlib.pyplot.Figure
        Plot handle containing the rendered 2D histogram.

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
    fig, ax = plt.subplots()
    _, _, _, hist_image = ax.hist2d(
        data.sel(space=x_coord), data.sel(space=y_coord), **kwargs
    )  # counts, xedges, yedges, image
    colourbar = fig.colorbar(hist_image, ax=ax)
    colourbar.solids.set(alpha=1.0)

    # Axis labels and title
    if not title and title_components:
        title = "Occupancy of " + ", ".join(title_components)
    if title:
        ax.set_title(title)
    ax.set_xlabel(x_coord)
    ax.set_ylabel(y_coord)

    return fig
