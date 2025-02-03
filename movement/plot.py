"""Wrappers to plot movement data."""

import xarray as xr
from matplotlib import pyplot as plt

DEFAULT_ARROW_ARGS = {
    "scale": 1.05,
    "headwidth": 3,
    "headlength": 4,
    "headaxislength": 4,
    "color": "gray",
}

DEFAULT_SCATTER_ARGS = {
    "s": 50,
    "cmap": "viridis",
}


def vector(
    ds: xr.DataArray,
    individual: int | str = 0,
    x_lim: tuple[float, float] | None = None,
    y_lim: tuple[float, float] | None = None,
    time_points: tuple[int, ...] | None = None,
    reference_points: list[str] | tuple[str, str] = (
        "left_ear",
        "right_ear",
    ),  # immutable tuple
    vector_point: str = "snout",
    title: str | None = None,
    **kwargs,
) -> plt.Figure:
    """Plot head vector for a specified subset of the data.

    Parameters
    ----------
    ds : xr.DataArray
        The dataset containing the movement data.
    individual : int | str, optional
        The individual to plot the vector for. Can be either the index
        of the individual in the dataset or the label of the individual.
        Default is 0.
    x_lim : tuple[float] | None, optional
        The x-axis limits for the plot. If None, the limits are set to
        the minimum and maximum x-values of the reference individual.
        Default is None.
    y_lim : tuple[float] | None, optional
        The y-axis limits for the plot. If None, the limits are set to
        the minimum and maximum y-values of the reference individual.
        Default is None.
    time_points : tuple[int] | None, optional
        The time points to plot the vector for, specified as a tuple
        containing the starting and ending frame. If None, the first
        15 frames with non-NaN space coordinates for the keypoints of
        interest are selected.
    reference_points : list[str] | tuple[str, str], optional
        The reference points to calculate the vector from. Default is
        ("left_ear", "right_ear").
    vector_point : str, optional
        The point to plot the vector for. Default is "snout".
    title : str | None, optional
        The title of the plot. If None, a title is generated using the
        name of the vector point and the individual. Default is None.
    **kwargs : dict
        Additional keyword arguments to pass to `plt.quiver` to create
        the arrows in the vector plot.


    Returns
    -------
    plt.Figure
        The figure containing the plot.

    """
    individual = (
        ds.individuals.values[individual]
        if isinstance(individual, int)
        else str(individual)
    )

    keypoints_of_interest = list(reference_points) + [vector_point]

    ds_sel = ds.sel(individuals=individual, keypoints=keypoints_of_interest)

    valid_time_points = ds_sel.position.dropna(
        dim="time", how="any"
    ).time.values
    time_points = time_points or valid_time_points[:15]

    position = ds_sel.position.sel(time=list(time_points))

    x_lim = x_lim or (
        position.sel(space="x").min(),
        position.sel(space="x").max(),
    )
    y_lim = y_lim or (
        position.sel(space="y").min(),
        position.sel(space="y").max(),
    )

    reference = position.sel(keypoints=list(reference_points)).mean(
        dim="keypoints"
    )
    vector = position.sel(keypoints=vector_point) - reference
    vector = vector.drop_vars("keypoints")

    arrow_kwargs = {
        key: kwargs.get(key, value)
        for key, value in DEFAULT_ARROW_ARGS.items()
    }
    scatter_kwargs = {
        key: kwargs.get(key, value)
        for key, value in DEFAULT_SCATTER_ARGS.items()
    }

    fig, ax = plt.subplots(1, 1)

    # Plot midpoint between the reference points
    sc = ax.scatter(
        reference.sel(space="x"),
        reference.sel(space="y"),
        c=time_points,
        marker="*",
    )

    # plot vector point
    ax.scatter(
        position.sel(
            space="x",
            keypoints=vector_point,
        ),
        position.sel(
            space="y",
            keypoints=vector_point,
        ),
        label=vector_point,
        c=time_points,
        marker="o",
        **scatter_kwargs,
    )

    # Plot vector of the vector point relative
    # to the midpoint between the reference points
    ax.quiver(
        reference.sel(space="x"),
        reference.sel(space="y"),
        vector.sel(space="x"),
        vector.sel(space="y"),
        angles="xy",
        scale_units="xy",
        **arrow_kwargs,
    )

    ax.axis("equal")
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.set_title(title or f"{vector_point} vector ({individual})")
    ax.invert_yaxis()
    colorbar_label = f"time ({position.attrs.get('time_unit', 'frames')})"
    fig.colorbar(
        sc, ax=ax, label=colorbar_label, ticks=list(time_points)[0::2]
    )

    ax.legend(
        [
            f"midpoint ({reference_points[0]}, {reference_points[1]})",
            f"{vector_point}",
            "vector",
        ],
        loc="best",
    )

    return fig


# # # FOR TESTING
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

# # head_trajectory = vector(
# #     ds,
# #     reference_points=("left_ear", "right_ear"),
# #     vector_point="snout",
# #     time_points=None,
# #     x_lim=None,
# #     y_lim=None,
# #     individual=0,
# # )

# # plt.ion()
# # head_trajectory.show()
# # # user input to close window
# # input("Press Enter to continue...")


# # area of interest
# xmin, ymin = 600.0, 665.0  # pixels
# x_delta, y_delta = 125.0, 100.0  # pixels

# # time window
# time_window = tuple(range(1650, 1671))  # tuple of frame numbers

# fig_head_vector = vector(
#     ds,
#     reference_points=["left_ear", "right_ear"],
#     vector_point="snout",
#     individual="individual_0",
#     x_lim=(xmin, xmin + x_delta),
#     y_lim=(ymin, ymin + y_delta),
#     time_points=time_window,
#     title="Zoomed in head vector (individual_0)",
# )

# fig_head_vector.show()
# input("Press Enter to continue...")
