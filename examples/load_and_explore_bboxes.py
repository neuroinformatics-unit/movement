"""Inspect crab trajectories"""

# %%
import ast
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import cycler

from movement.io import load_bboxes

# %matplotlib widget

# %%%%%%%%%%%%%%%%%%%%%
# input data
file_csv = (
    "/Users/sofia/arc/project_Zoo_crabs/escape_clips/"
    "crabs_track_output_selected_clips/04.09.2023-04-Right_RE_test/predicted_tracks.csv"
)


# load ground truth!
groundtruth_csv = (
    "/Users/sofia/arc/project_Zoo_crabs/escape_clips/"
    "04.09.2023-04-Right_RE_test_corrected_ST_csv_SM.csv"
)

# %%%%%%%%%%%%%%%%%%%%%%%%%
# Fix ground truth file
df = pd.read_csv(groundtruth_csv, sep=",", header=0)

# find duplicates
list_unique_filenames = list(set(df.filename))
filenames_to_rep_ID = {}
for file in list_unique_filenames:
    df_one_filename = df.loc[df["filename"] == file]

    list_track_ids_one_filename = [
        int(ast.literal_eval(row.region_attributes)["track"])
        for row in df_one_filename.itertuples()
    ]

    if len(set(list_track_ids_one_filename)) != len(
        list_track_ids_one_filename
    ):
        # [
        #     list_track_ids_one_filename.remove(k)
        #     for k in set(list_track_ids_one_filename)
        # ]  # there could be more than one duplicate!!!
        for k in set(list_track_ids_one_filename):
            list_track_ids_one_filename.remove(k)  # remove first occurrence

        filenames_to_rep_ID[file] = list_track_ids_one_filename

# delete duplicate rows
for file, list_rep_ID in filenames_to_rep_ID.items():
    for rep_ID in list_rep_ID:
        # find repeated rows for selected file and rep_ID
        matching_rows = df[
            (df["filename"] == file)
            & (df["region_attributes"] == f'{{"track":"{rep_ID}"}}')
        ]

        # Identify the index of the first matching row
        if not matching_rows.empty:
            indices_to_drop = matching_rows.index[1:]

            # Drop all but the first matching row
            df = df.drop(indices_to_drop)

# save to csv
groundtruth_csv_corrected = Path(groundtruth_csv).parent / Path(
    Path(groundtruth_csv).stem + "_corrected.csv"
)
df.to_csv(groundtruth_csv_corrected, index=False)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Read corrected ground truth as movement dataset
ds_gt = load_bboxes.from_via_tracks_file(
    groundtruth_csv_corrected, fps=None, use_frame_numbers_from_file=False
)
print(ds_gt)

# Print summary
print(f"{ds_gt.source_file}")
print(f"Number of frames: {ds_gt.sizes['time']}")
print(f"Number of individuals: {ds_gt.sizes['individuals']}")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Read predictions as movement dataset
ds_pred = load_bboxes.from_via_tracks_file(
    file_csv, fps=None, use_frame_numbers_from_file=False
)
print(ds_pred)

# Print summary
print(f"{ds_pred.source_file}")
print(f"Number of frames: {ds_pred.sizes['time']}")
print(f"Number of individuals: {ds_pred.sizes['individuals']}")


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Check when individuals are labelled
# check x and y coordinates are nan at the same locations
# TODO: change colormap to white and blue
assert (
    np.isnan(ds_gt.position.data[:, :, 0])
    == np.isnan(ds_gt.position.data[:, :, 1])
).all()

fig, axs = plt.subplots(2, 1, sharex=True)

axs[0].matshow(np.isnan(ds_gt.position.data[:, :, 0]).T, aspect="auto")
axs[0].set_title("Ground truth")
axs[0].set_xlabel("time (frames)")
axs[0].set_ylabel("individual")

axs[1].matshow(np.isnan(ds_pred.position.data[:, :, 0]).T, aspect="auto")
axs[1].set_title("Prediction")
axs[1].set_xlabel("time (frames)")
axs[1].set_ylabel("tracks")
axs[1].xaxis.tick_bottom()

# # add reference
# axs[1].hlines(
#     y=ds_gt.sizes["individuals"],
#     xmin=0,
#     xmax=ds_gt.sizes["time"] - 1,
#     color="red",
# )

fig.subplots_adjust(hspace=0.6, wspace=0.5)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Compare lengths of continuous tracks & plot distrib

# for each individual, find the length of chunks between nans
map_individuals_to_chunks = {}
for individual in range(ds_gt.sizes["individuals"]):
    # find nans in x-coord for that individual
    nan_idx = np.isnan(ds_gt.position.data[:, individual, 0])

    # find lengths of continuous tracks
    len_chunks = [
        len(list(group_iter))
        for key, group_iter in itertools.groupby(nan_idx)
        if not key
    ]

    map_individuals_to_chunks[individual] = len_chunks

# %%
fig, ax = plt.subplots(1, 1)
for ind, list_chunks in map_individuals_to_chunks.items():
    ax.scatter([ind] * len(list_chunks), list_chunks)


# [sum(1 for _ in input) for _, input in itertools.groupby(_)]

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Check confidence of detections
confidence_values = ds_pred.confidence.data.flatten()
nan_median_confidence = np.nanmedian(confidence_values)


fig, ax = plt.subplots(1, 1)
hist = ax.hist(confidence_values, bins=np.arange(0, 1.01, 0.05))
ax.vlines(x=nan_median_confidence, ymin=0, ymax=max(hist[0]), color="red")
ax.set_aspect("auto")

fig, ax = plt.subplots(1, 1)
ax.hist(ds_pred.confidence.data.flatten(), bins=np.arange(0.6, 1.01, 0.01))
ax.vlines(x=nan_median_confidence, ymin=0, ymax=max(hist[0]), color="red")
ax.set_aspect("auto")

print(f"Median confidence: {nan_median_confidence}")

# %%
# plot all trajectories
# ds.position ---> time, individuals, space
# why noise? remove low predictions?

for ds, title in zip(
    [ds_gt, ds_pred], ["Ground truth", "Prediction"], strict=False
):
    # cmap = plt.get_cmap('tab10')
    fig, ax = plt.subplots(1, 1)
    plt.rcParams["axes.prop_cycle"] = cycler(
        color=plt.get_cmap("tab10").colors
    )

    for ind_idx in range(ds.sizes["individuals"]):
        ax.scatter(
            x=ds.position[:, ind_idx, 0],  # nframes, nindividuals, x
            y=ds.position[:, ind_idx, 1],
            s=1,
            # c=cmap(ind_idx),
        )
    ax.set_aspect("equal")
    ax.set_ylim(-150, 2500)
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.set_title(title)
    plt.show()

# %%
# first 10 individuals
fig, ax = plt.subplots(1, 1)

ax.scatter(x=ds_pred.position[:, :10, 0], y=ds_pred.position[:, :10, 1], s=1)
ax.set_aspect("equal")
ax.set_xlabel("x (pixels)")
ax.set_ylabel("y (pixels)")
# %%
# groupby
# It generates a break or new group every time the value of the key function
# changes
# input = (
#   np.isnan(ds_gt.position.data[:,0,0]*ds_gt.position.data[:,0,1]
#  ).astype(int))
input = [0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1]
len_per_chunk = [
    (key, len(list(group_iter)))
    for key, group_iter in itertools.groupby(input)
]
len_per_chunk_with_1 = [
    len(list(group_iter))
    for key, group_iter in itertools.groupby(input)
    if key == 1
]
