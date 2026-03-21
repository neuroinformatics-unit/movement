"""Wrappers for plotting occupancy data of select individuals, with enhanced options."""
from collections.abc import Hashable, Sequence
from typing import Any, Literal, TypeAlias, Union
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure
HistInfoKeys: TypeAlias = Literal["h", "xedges", "yedges"]
DEFAULT_HIST_ARGS = {"alpha": 1.0, "bins": 30, "cmap": "viridis"}
def plot_occupancy(
    da: xr.DataArray,
    individuals: Union[Hashable, Sequence[Hashable]] | None = None,
    keypoints: Union[Hashable, Sequence[Hashable]] | None = None,
    ax: Axes | None = None,
    normalize: bool = False,
    return_centroid: bool = False,
    auto_range: bool = True,
    **kwargs: Any,
) -> tuple[Figure | SubFigure, Axes, dict[HistInfoKeys, np.ndarray], xr.DataArray | None]:
    """
    Create a 2D occupancy histogram of positions with flexible plotting options.
    Features:
    - Handles multiple keypoints (plots centroid).
    - Aggregates over multiple individuals.
    - Can normalize counts.
    - Can return centroid data for further analysis.
    - Flexible binning and range handling.
    Parameters
    ----------
    da
        Spatial data to create histogram for. NaN values are dropped.
    individuals
        Name(s) of individuals to plot. By default, all individuals are aggregated.
    keypoints
        Name(s) of keypoints. Centroid of all selected keypoints is computed.
    ax
        Axes object to plot on. If None, a new figure and axes are created.
    normalize
        If True, histogram counts are normalized to sum to 1.
    return_centroid
        If True, return the centroid (aggregated) data as a DataArray.
    auto_range
        If True, automatically compute bin edges based on min/max of data.
    kwargs
        Passed to matplotlib's hist2d function.
    Returns
    -------
    fig : Figure | SubFigure
        Figure containing the histogram.
    ax : Axes
        Axes on which the histogram was drawn.
    hist2d_info : dict[Literal['h', 'xedges', 'yedges'], np.ndarray]
        Histogram counts and bin edges.
    centroid_data : xr.DataArray | None
        The computed centroid data if return_centroid=True, else None.
    """
    data = da.copy(deep=True)
    if "keypoints" in data.dims and keypoints is not None:
        data = data.sel(keypoints=keypoints)
    if "keypoints" in data.dims:
        data = data.mean(dim="keypoints", skipna=True)
    if "individuals" in data.dims and individuals is not None:
        data = data.sel(individuals=individuals)
    if "individuals" in data.dims:
        data = data.stack(temp=("time", "individuals")).swap_dims({"temp": "time"})
    data = data.dropna(dim="time", how="any")
    centroid_data = data if return_centroid else None
    x_coord, y_coord = data["space"].values[:2]
    for key, value in DEFAULT_HIST_ARGS.items():
        kwargs.setdefault(key, value)
    if auto_range and "range" not in kwargs:
        x_min, x_max = data.sel(space=x_coord).min().item(), data.sel(space=x_coord).max().item()
        y_min, y_max = data.sel(space=y_coord).min().item(), data.sel(space=y_coord).max().item()
        kwargs["range"] = [[x_min, x_max], [y_min, y_max]]
    fig = ax.get_figure() if ax is not None else None
    if fig is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    h, xedges, yedges, hist_image = ax.hist2d(
        data.sel(space=x_coord), data.sel(space=y_coord), **kwargs
    )
    if normalize:
        h = h / h.sum()
        hist_image.set_array(h.ravel())
    cbar = fig.colorbar(hist_image, ax=ax)
    if cbar.solids is not None:
        cbar.solids.set(alpha=1.0)
    ax.set_xlabel(str(x_coord))
    ax.set_ylabel(str(y_coord))
    ax.set_title("Occupancy")
    hist2d_info: dict[HistInfoKeys, np.ndarray] = {
        "h": h,
        "xedges": xedges,
        "yedges": yedges,
    }
    return fig, ax, hist2d_info, centroid_data
