"""Wrappers to plot movement data."""

from pathlib import Path

import numpy as np
import xarray as xr
from matplotlib import pyplot as plt


def trajectory(
    ds: xr.DataArray,
    keypoint: str | list[str],
    individual: int | str = 0,
    frame_path: None | Path = None,
) -> plt.Figure:
    """Plot trajectory.

    This function plots the  trajectory of a specified keypoint or the midpoint
    between two keypoints for a given individual. The individual can be
    specified by their index or name. By default, the first individual
    is selected.
    """
    position = ds.position
    if isinstance(keypoint, list):
        midpoint = position.sel(keypoints=keypoint).mean(dim="keypoints")
        midpoint_name = "midpoint between " + " and ".join(keypoint)
    elif isinstance(keypoint, str):
        midpoint = position.sel(keypoints=keypoint)
        midpoint_name = keypoint

    if isinstance(individual, int):
        individual = ds.individuals.values[individual]
    elif isinstance(individual, str):
        individual = np.str_(individual)

    fig, ax = plt.subplots(1, 1)

    sc = ax.scatter(
        midpoint.sel(individuals=individual, space="x"),
        midpoint.sel(individuals=individual, space="y"),
        s=15,
        c=midpoint.time,
        cmap="viridis",
        marker="o",
    )

    ax.axis("equal")
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.invert_yaxis()
    ax.set_title(f"{individual} trajectory of {midpoint_name}")
    fig.colorbar(sc, ax=ax, label=f"time ({ds.attrs['time_unit']})")

    if frame_path is not None:
        sc.set_alpha(0.05)
        frame = plt.imread(frame_path)
        ax.invert_yaxis()
        ax.imshow(frame)
        # Ivert the y-axis back again since the the image is plotted
        # using a coordinate system with origin on the top left of the image

    return fig
