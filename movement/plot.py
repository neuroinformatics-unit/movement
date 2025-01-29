"""Wrappers to plot movement data."""

from pathlib import Path

import numpy as np
import xarray as xr
from matplotlib import pyplot as plt

DEFAULT_PLOTTING_ARGS = {
    "s": 15,
    "cmap": "viridis",
    "marker": "o",
    "alpha": 0.5,
}


def trajectory(
    ds: xr.DataArray,
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
    """
    position = ds.position
    if isinstance(keypoint, list):
        plotting_point = position.sel(keypoints=keypoint).mean(dim="keypoints")
        plotting_point_name = "midpoint between " + " and ".join(keypoint)
    elif isinstance(keypoint, str):
        plotting_point = position.sel(keypoints=keypoint)
        plotting_point_name = keypoint

    if isinstance(individual, int):
        # Index-based individual reference
        individual = ds.individuals.values[individual]
    elif isinstance(individual, str):
        # Label-based individual reference
        individual = np.str_(individual)

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
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title(f"{individual} trajectory of {plotting_point_name}")
    fig.colorbar(sc, ax=ax, label=f"time ({ds.attrs['time_unit']})")

    if frame_path is not None:
        frame = plt.imread(frame_path)
        # Ivert the y-axis back again since the the image is plotted
        # using a coordinate system with origin on the top left of the image
        ax.invert_yaxis()
        ax.imshow(frame)

    return fig


# FOR TESTING
# from movement import sample_data
# from movement.io import load_poses

# ds_path = sample_data.fetch_dataset_paths(
#     "SLEAP_single-mouse_EPM.analysis.h5"
# )["poses"]
# ds = load_poses.from_sleap_file(ds_path, fps=None)
# # force time_unit = frames
# frame_path = sample_data.fetch_dataset_paths(
#     "SLEAP_single-mouse_EPM.analysis.h5"
# )["frame"]

# head_trajectory = trajectory(
#     ds,
#     ["left_ear", "right_ear"],
#     individual=0,
#     frame_path=frame_path,
#     s=10,  # We could leave these options off
#     cmap="viridis",  # if we include the default
#     marker="o",  # argument setting-block above
#     alpha=0.05,
#     title="Head trajectory of individual 0",
# )

# plt.ion()
# head_trajectory.show()
# # user input to close window
# input("Press Enter to continue...")
