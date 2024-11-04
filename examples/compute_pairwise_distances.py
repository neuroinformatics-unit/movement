# ruff: noqa: E402
"""Compute distances between keypoints.
====================================

Compute pairwise distances between keypoints, within and across individuals.
"""

# %%
# Imports
# -------

import numpy as np

# For interactive plots: install ipympl with `pip install ipympl` and uncomment
# the following line in your notebook
# %matplotlib widget
from matplotlib import pyplot as plt

from movement import sample_data
from movement.kinematics import compute_pairwise_distances

# %%
# Load sample dataset
# ------------------------
# First, we load an example dataset. In this case, we select the
# ``DLC_two-mice.predictions.csv`` sample data.
ds = sample_data.fetch_dataset(
    "DLC_two-mice.predictions.csv",
)

print(ds)

# 2 individuals, 12 keypoints, 2d, time in seconds, 59999 frames

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Visually inspect the data on top of the sample frame

# compute centroid of all keypoints
ds["centroid"] = ds.position.mean(dim="keypoints")

# read sample frame
im = plt.imread(ds.frame_path)

fig, ax = plt.subplots()
ax.imshow(im)
for ind in ds.coords["individuals"].data:
    ax.scatter(
        x=ds.centroid.sel(individuals=ind, space="x"),
        y=ds.centroid.sel(individuals=ind, space="y"),
        s=5,
        label=f"{ind}",
        alpha=0.05,
        # color=cmap(i),
    )
ax.set_xlabel("x (pixels)")
ax.set_ylabel("y (pixels)")
ax.legend()


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Get reference length

# Should I use diagonal?
start_point = np.array([[209, 382]])
end_point = np.array([[213, 1022]])

reference_length = np.linalg.norm(end_point - start_point)

fig, ax = plt.subplots()
ax.imshow(im)
ax.plot(
    [start_point[:, 0], end_point[:, 0]],
    [start_point[:, 1], end_point[:, 1]],
    "r",
)
ax.text(
    1.01 * (start_point[0, 0] + end_point[0, 0]),
    0.49 * (start_point[0, 1] + end_point[0, 1]),
    f"{reference_length:.2f} pixels",
    color="r",
    horizontalalignment="center",
)
ax.set_xlabel("x (pixels)")
ax.set_ylabel("y (pixels)")
ax.set_title("Reference length")

# Measure the long side of the box in pixels

# Note the lens is a bit distorted


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Plot keypoints on sample frames

# time_sel = [5.0, 50.0, 100.0]

# # get colormap tab20
# cmap = plt.get_cmap("tab20")

# fig, axs = plt.subplots(len(time_sel), 1, figsize=(10, 15))
# for k in range(len(time_sel)):
#     for kpt_i, kpt in enumerate(ds.coords["keypoints"].data):
#         axs[k].scatter(
#             x=ds.position.sel(keypoints=kpt, space="x", time=time_sel[k]),
#             y=ds.position.sel(keypoints=kpt, space="y", time=time_sel[k]),
#             s=10,
#             label=f"{kpt}",
#             color=cmap(kpt_i),
#         )

#     axs[k].axis("equal")
#     axs[k].invert_yaxis()
#     axs[k].set_xlabel("x (pixels)")
#     axs[k].set_ylabel("y (pixels)")
#     axs[k].set_title(f"Keypoints at {time_sel[k]} s")
#     # axs[k].legend(bbox_to_anchor=(1.1, 1.05))


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Compute distances between keypoints on different individuals

inter_individual_kpt_distances = compute_pairwise_distances(
    ds.position,
    dim="individuals",
    pairs={
        "individual1": "individual2",
        # this will set the dims of the output,
        # (keypoints will be the coordinates)
    },
)  # pixels, dimensions are individual1 and individual2

# for each frame, this matrix has the distance between all keypoints
# from individual 1 to all keypoints on individual 2
print(inter_individual_kpt_distances.shape)  # inter_individual_distances

# normalise with reference length?
inter_individual_kpt_distances_norm = (
    inter_individual_kpt_distances / reference_length
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Plot matrix of distances and keypoints
# Show different patterns / positions between the two animals
# Note that the colorbars vary across frames!

time_sel = [50.0, 100.0, 250.0]

# get colormap tab20 for keypoints
cmap = plt.get_cmap("tab20")

# get list of keypoints per individual
# (it may not be the same)
list_kpts_individual_1 = list(
    inter_individual_kpt_distances.coords["individual1"].data
)
list_kpts_individual_2 = list(
    inter_individual_kpt_distances.coords["individual2"].data
)

for k in range(len(time_sel)):
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    fig.subplots_adjust(wspace=0.5)

    # plot keypoints
    for kpt_i, kpt in enumerate(ds.coords["keypoints"].data):
        axs[0].scatter(
            x=ds.position.sel(keypoints=kpt, space="x", time=time_sel[k]),
            y=ds.position.sel(keypoints=kpt, space="y", time=time_sel[k]),
            s=10,
            label=f"{kpt}",
            color=cmap(kpt_i),
        )
        # # connect matching keypoints
        # axs[0].plot(
        #     ds.position.sel(keypoints=kpt, space="x", time=time_sel[k]),
        #     ds.position.sel(keypoints=kpt, space="y", time=time_sel[k]),
        #     'r'
        # )

    # add text per individual
    for ind in ds.coords["individuals"].data:
        axs[0].text(
            ds.centroid.sel(individuals=ind, space="x", time=time_sel[k]),
            ds.centroid.sel(individuals=ind, space="y", time=time_sel[k]),
            ind,
            horizontalalignment="left",
            # verticalalignment="center",
        )
    axs[0].invert_yaxis()
    axs[0].set_xlabel("x (pixels)")
    axs[0].set_ylabel("y (pixels)")
    axs[0].set_title(f"Keypoints at {time_sel[k]} s")
    axs[0].axis("equal")
    axs[0].legend()  # bbox_to_anchor=(1.1, 1.05))

    # plot distances normalised matrix
    im = axs[1].imshow(
        inter_individual_kpt_distances.sel(time=time_sel[k]),
        # vmin=0,
        # vmax=1,
    )
    axs[1].set_xticks(range(0, len(list_kpts_individual_1)))
    axs[1].set_yticks(range(0, len(list_kpts_individual_2)))
    axs[1].set_xticklabels(
        inter_individual_kpt_distances.coords["individual1"].data,
        rotation=45,
    )
    axs[1].set_yticklabels(
        inter_individual_kpt_distances.coords["individual2"].data,
        rotation=0,
    )

    axs[1].set_xlabel(inter_individual_kpt_distances.dims[1])
    axs[1].set_ylabel(inter_individual_kpt_distances.dims[2])
    axs[1].set_title(f"Inter-individual keypoint distances at {time_sel[k]} s")

    # cbar = plt.colorbar(im, ax=axs[0])
    # cbar.set_label("distance (pixels)")
    fig.colorbar(
        im,
        ax=axs[1],
        label="distance (pixels)",
        # use_gridspec=True
        # ticks=np.linspace(0,1,5)
        # ticks=list(range(0, int(reference_length), 100)),
    )


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# To get distance between homologous keypoints
# get the diagonal of the previous matrix at each frame
inter_individual_same_kpts = np.diagonal(inter_individual_kpt_distances)
print(inter_individual_same_kpts.shape)  # (59999, 12)


# should match selecting each keypoint manually
for k_i, kpt in enumerate(list_kpts_individual_1):
    np.testing.assert_almost_equal(
        inter_individual_kpt_distances.sel(individual1=kpt, individual2=kpt),
        inter_individual_same_kpts[:, k_i],
    )


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# To get specific keypoints between individuals
# e.g. snout of individual 1 to tail base of individual 2
# yo can select the relevant keypoints along the dimensions
# "individual1" and "individual2"

distance_snout_1_to_tail_2 = inter_individual_kpt_distances.sel(
    individual1="snout", individual2="tailbase"
)

# plot distance over time
fig, ax = plt.subplots()
ax.plot(
    distance_snout_1_to_tail_2.time,  # seconds
    distance_snout_1_to_tail_2 / reference_length,
)
ax.set_xlabel("time (seconds)")
ax.set_ylabel("distance normalised")

# plot vectors on top of a given frame?
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Compute distances between the keypoints of the same individual
# compute average bodylength = snout to tailbase

distance_snout_to_tailbase_all = compute_pairwise_distances(
    ds.position,
    dim="keypoints",
    pairs={
        "snout": "tailbase",
        # this will set the dims of the output (individuals will be the
        # coordinates)
    },
)  # pixels

print(distance_snout_to_tailbase_all)  # dimensions are snout and tailbase!

# within individual
bodylength_individual_1 = distance_snout_to_tailbase_all.sel(
    snout="individual1",
    tailbase="individual1",
)

bodylength_individual_2 = distance_snout_to_tailbase_all.sel(
    snout="individual2",
    tailbase="individual2",
)

# across individuals
# (an alternative way to the above)
snout_1_to_tail_2 = distance_snout_to_tailbase_all.sel(
    snout="individual1",
    tailbase="individual2",
)
snout_2_to_tail_1 = distance_snout_to_tailbase_all.sel(
    snout="individual2",
    tailbase="individual1",
)

np.testing.assert_almost_equal(
    snout_1_to_tail_2.data, distance_snout_1_to_tail_2.data
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Plot bodylength over time

for b_i, bodylength_data_array in enumerate(
    [
        bodylength_individual_1,
        bodylength_individual_2,
    ]
):
    fig, ax = plt.subplots()
    ax.plot(
        bodylength_data_array.time,
        bodylength_data_array,
    )
    ax.hlines(
        bodylength_data_array.mean(dim="time"),
        bodylength_data_array.time.min(),
        bodylength_data_array.time.max(),
        "r",
        label="mean length",
    )
    ax.set_title(f"Bopdy length of individual {b_i+1}")
    ax.set_xlabel("time (seconds)")
    ax.set_ylabel("length (pixels)")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Try usage of 'all' and plot matrix of four quadrants

# %%
# Try a different metric
