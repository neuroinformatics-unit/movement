"""Wrappers to plot movement data."""

from pathlib import Path

import numpy as np
import xarray as xr
from matplotlib import pyplot as plt

from movement import sample_data
from movement.io import load_poses


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
    if frame_path is not None:
        frame = plt.imread(frame_path)
        ax.imshow(frame)

    sc = ax.scatter(
        midpoint.sel(individuals=individual, space="x"),
        midpoint.sel(individuals=individual, space="y"),
        s=15,
        c=midpoint.time,
        cmap="viridis",
        marker="o",
    )

    if frame_path is not None:
        sc.set_alpha(0.5)

    ax.axis("equal")
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.invert_yaxis()
    ax.set_title(f"{individual} trajectory of {midpoint_name}")
    fig.colorbar(sc, ax=ax, label=f"time ({ds.attrs['time_unit']})")

    return fig


ds_path = sample_data.fetch_dataset_paths(
    "SLEAP_single-mouse_EPM.analysis.h5"
)["poses"]
ds = load_poses.from_sleap_file(ds_path, fps=None)

frame_path = sample_data.fetch_dataset_paths(
    "SLEAP_single-mouse_EPM.analysis.h5"
)["frame"]


head_trajectory = trajectory(
    ds, ["left_ear", "right_ear"], individual=0, frame_path=frame_path
)

plt.ion()  # Enable interactive mode
head_trajectory.show()
input("Press Enter to exit...")
