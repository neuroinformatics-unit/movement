"""Wrappers to plot movement data."""

import xarray as xr
from matplotlib import pyplot as plt

DEFAULT_SCATTER_ARGS = {
    "s": 15,
    "cmap": "viridis",
    "alpha": 1.0,
}

DEFAULT_ARROW_ARGS = {
    "scale": 1.05,
    "headwidth": 3,
    "headlength": 4,
    "headaxislength": 4,
    "color": "gray",
}


def _calculate_centroid(da):
    centroid = (
        da.mean(dim="keypoints", skipna=True).squeeze()
        if "keypoints" in da.dims and da.sizes["keypoints"] > 1
        else da.squeeze()
    )
    return centroid


def plot_vector(
    da: xr.DataArray,
    reference_keypoints: str | list[str] | None = None,
    vector_keypoints: str | list[str] | None = None,
    individual: str | None = None,
    ax: plt.Axes | None = None,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot vector.

    This function plots arrows from the centroid of the reference keypoint(s)
    to the centroid of the vector keypoint()s for a given individual. By
    default, the first individual is selected.

    Parameters
    ----------
    da : xr.DataArray
        A data array containing position information, with `time` and `space`
        as required dimensions. Optionally, it may have `individuals` and/or
        `keypoints` dimensions.
    individual : str, optional
        The name of the individual to be plotted. By default, the first
        individual is plotted.
    reference_keypoints : str, list[str], optional
        The reference keypoints used to calculate the centroid from which the
        arrows are plotted. By default, the centroid of all keypoints is used
        as the reference.
    vector_keypoints : str, list[str], optional
        The keypoint(s) used to calculate the centroid to which the arrows are
        plotted. By default the first keypoint is used as vector keypoint.
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

    if "keypoints" not in da.dims:
        raise ValueError(
            "DataArray must have 'keypoints' dimension to plot vectors."
        )

    selection = {}
    if "individuals" in da.dims and individual is None:
        selection = {"individuals": da.individuals.values[0]}

    title_suffix = (
        f" of {individual}"
        if "individuals" in da.dims and individual is not None
        else ""
    )

    if "keypoints" in da.dims and reference_keypoints is None:
        reference_keypoints = da.keypoints.values

    if "keypoints" in da.dims and vector_keypoints is None:
        vector_keypoints = da.keypoints.values[0]

    reference_centroid = _calculate_centroid(
        da.sel(keypoints=reference_keypoints, **selection)
    )
    vector_centroid = _calculate_centroid(
        da.sel(keypoints=vector_keypoints, **selection)
    )

    vector = vector_centroid - reference_centroid

    fig, ax = plt.subplots(figsize=(6, 6)) if ax is None else (ax.figure, ax)

    # Merge default plotting args with user-provided kwargs
    arrow_kwargs = {
        key: kwargs.get(key, value)
        for key, value in DEFAULT_ARROW_ARGS.items()
    }
    scatter_kwargs = {
        key: kwargs.get(key, value)
        for key, value in DEFAULT_SCATTER_ARGS.items()
    }

    colorbar = False
    if "c" not in kwargs:
        kwargs["c"] = da.time
        colorbar = True

    # Plot the scatter, of the two centroids
    sc = ax.scatter(
        reference_centroid.sel(space="x"),
        reference_centroid.sel(space="y"),
        marker="*",
        c=kwargs["c"],
        **scatter_kwargs,
    )
    ax.scatter(
        vector_centroid.sel(space="x"),
        vector_centroid.sel(space="y"),
        marker="o",
        c=kwargs["c"],
        **scatter_kwargs,
    )

    # Plot the arrows
    ax.quiver(
        reference_centroid.sel(space="x"),
        reference_centroid.sel(space="y"),
        vector.sel(space="x"),
        vector.sel(space="y"),
        angles="xy",
        scale_units="xy",
        **arrow_kwargs,
    )

    space_unit = da.attrs.get("space_unit", "pixels")
    ax.set_xlabel(f"x ({space_unit})")
    ax.set_ylabel(f"y ({space_unit})")
    ax.axis("equal")
    ax.set_title(f"Vector {title_suffix}")

    # Add 'colorbar' for time dimension if no colour was provided by user
    time_unit = da.attrs.get("time_unit")
    time_label = f"time ({time_unit})" if time_unit else "time steps (frames)"
    fig.colorbar(sc, ax=ax, label=time_label).solids.set(
        alpha=1.0
    ) if colorbar else None

    ax.legend(
        [
            f"reference: {reference_keypoints}",
            f"{vector_keypoints}",
            "vector",
        ],
        loc="best",
    )

    return fig, ax
