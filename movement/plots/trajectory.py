"""Wrappers to plot movement data."""

from contextlib import suppress
from typing import Any, cast

import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure, SubFigure

DEFAULT_PLOTTING_ARGS = {
    "s": 15,
    "marker": "o",
    "alpha": 1.0,
}


def plot_centroid_trajectory(  # noqa: C901
    da: xr.DataArray,
    individual: str | None = None,
    keypoints: str | list[str] | None = None,
    ax: Axes | None = None,
    **kwargs: Any,
) -> tuple[Figure | SubFigure, Axes]:
    """Plot centroid trajectory.

    Parameters
    ----------
    da : xr.DataArray
        Position data with ``time`` and ``space`` dimensions.
    individual : str, optional
        Individual to plot.
    keypoints : str or list[str], optional
        Keypoint(s) whose centroid will be plotted.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.

    Returns
    -------
    (figure, axes) : tuple of (
        matplotlib.figure.Figure | SubFigure,
        matplotlib.axes.Axes,
    )
        The figure and axes containing the trajectory plot.

    """
    if isinstance(individual, list):
        raise ValueError("Only one individual can be selected.")

    selection: dict[str, Any] = {}

    if "individuals" in da.dims:
        selection["individuals"] = (
            da.individuals.values[0] if individual is None else individual
        )

    if "keypoints" in da.dims:
        selection["keypoints"] = (
            da.keypoints.values if keypoints is None else keypoints
        )

    plot_point = da.sel(**selection)

    if "keypoints" in plot_point.dims and plot_point.sizes["keypoints"] > 1:
        plot_point = plot_point.mean(dim="keypoints", skipna=True)

    plot_point = plot_point.squeeze()

    fig: Figure | SubFigure

    if ax is None:
        f, a = plt.subplots(figsize=(6, 6))
        fig = cast(Figure | SubFigure, f)
        ax = a
    else:
        _fig = ax.get_figure()
        if _fig is None:
            raise RuntimeError("Axes object is not associated with a Figure.")
        fig = cast(Figure | SubFigure, _fig)

    for key, value in DEFAULT_PLOTTING_ARGS.items():
        kwargs.setdefault(key, value)

    colorbar = False
    if "c" not in kwargs:
        kwargs["c"] = plot_point.time
        colorbar = True

    sc = ax.scatter(
        plot_point.sel(space="x"),
        plot_point.sel(space="y"),
        **kwargs,
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Trajectory")

    if colorbar:
        cb: Colorbar | None = None
        try:
            cb = fig.colorbar(sc, ax=ax, label="Time")
        except Exception:
            cb = None

        if cb is not None and cb.solids is not None:
            with suppress(Exception):
                cb.solids.set(alpha=1.0)

    return cast(Figure | SubFigure, fig), ax
