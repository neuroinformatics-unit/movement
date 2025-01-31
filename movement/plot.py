"""Wrappers to plot movement data."""

import numpy as np
import xarray as xr
from matplotlib import pyplot as plt

DEFAULT_PLOTTING_ARGS = {
    "s": 15,
    "cmap": "viridis",
    "marker": "o",
    "alpha": 0.5,
}


def vector(
    ds: xr.DataArray,
    individual: int | str = 0,
    x: tuple[int] | None = None,
    y: tuple[int] | None = None,
    time_range: range | None = None,
    reference_points: list[str] | tuple[str, str] = (
        "left_ear",
        "right_ear",
    ),  # immutable tuple
    vector_point: str = "snout",
    title: str | None = None,
    **kwargs,
) -> plt.Figure:
    """Plot head vector for a specified subset of the data."""
    position = ds.position
    reference = position.sel(keypoints=list(reference_points)).mean(
        dim="keypoints"
    )
    vector = position.sel(keypoints=vector_point) - reference
    vector = vector.drop_vars("keypoints")

    if isinstance(individual, int):
        individual = ds.individuals.values[individual]  # Index-based reference
    elif isinstance(individual, str):
        individual = np.str_(individual)  # Label-based reference

    if time_range is None:
        time_range = range(len(ds.time))

    for coord in ["x", "y"]:
        if locals()[coord] is None:
            locals()[coord] = (
                reference.sel(
                    individuals=individual, space=coord, time=time_range
                ).min(),
                reference.sel(
                    individuals=individual, space=coord, time=time_range
                ).max(),
            )

    fig, ax = plt.subplots(1, 1)

    # Plot midpoint between the reference points
    sc = ax.scatter(
        reference.sel(individuals=individual, space="x", time=time_range),
        reference.sel(individuals=individual, space="y", time=time_range),
        label="reference",
    )

    # plot vector point
    ax.scatter(
        position.sel(
            individuals=individual,
            space="x",
            time=time_range,
            keypoints=vector_point,
        ),
        position.sel(
            individuals=individual,
            space="y",
            time=time_range,
            keypoints=vector_point,
        ),
        label=vector_point,
    )

    # Plot vector of the vector point relative
    # to the midpoint between the reference points
    ax.quiver(
        reference.sel(individuals=individual, space="x", time=time_range),
        reference.sel(individuals=individual, space="y", time=time_range),
        vector.sel(individuals=individual, space="x", time=time_range),
        vector.sel(individuals=individual, space="y", time=time_range),
        angles="xy",
        scale=1,
        scale_units="xy",
        headwidth=7,
        headlength=9,
        headaxislength=9,
        color="gray",
    )

    ax.axis("equal")
    ax.set_xlim(x)
    ax.set_ylim(y)
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title(f"Head vector ({individual})")
    ax.invert_yaxis()
    fig.colorbar(
        sc,
        ax=ax,
        label=f"time ({ds.attrs['time_unit']})",
        ticks=list(time_range)[0::2],
    )

    ax.legend(
        [
            "reference",
            f"{vector_point}",
            "vector",
        ],
        loc="best",
    )

    return fig


# # FOR TESTING
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

# # area of interest
# xmin, ymin = 600, 665  # pixels
# x_delta, y_delta = 125, 100  # pixels

# # time window
# time_window = range(1650, 1671)  # frames

# # Plot the head vector for the specified subset of the data
# head_vector = vector(
#     ds,
#     individual=0,
#     x=(xmin, xmin + x_delta),
#     y=(ymin, ymin + y_delta),
#     time_range=time_window,
#     reference_points=("left_ear", "right_ear"),
#     vector_point="snout",
# )

# plt.ion()
# head_vector.show()
# input("Press Enter to continue...")
