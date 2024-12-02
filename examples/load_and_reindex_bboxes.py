"""Load and reindex bounding boxes tracks
==========================================

Load an example dataset of bounding boxes' tracks and reindex
it to every frame.
"""

# %%
# Imports
# -------

# For interactive plots: install ipympl with `pip install ipympl` and uncomment
# the following line in your notebook
# %matplotlib widget
import csv
import math
import os

import sleap_io as sio
from matplotlib import pyplot as plt

from movement import sample_data
from movement.filtering import interpolate_over_time
from movement.io import load_bboxes

# %%
# Load sample dataset
# ------------------------
# In this tutorial, we will use a sample bounding boxes dataset with
# a single individual (a crab). The clip is part of the `Moving
# Camouflaged Animals Dataset (MoCA) dataset <https://www.robots.ox.ac.uk/~vgg/data/MoCA/>`_.
#
# We will also download the associated video for visualising the data later.

dataset_dict = sample_data.fetch_dataset_paths(
    "VIA_single-crab_MOCA-crab-1.csv",
    with_video=True,  # download associated video
)

file_path = dataset_dict["bboxes"]
print(file_path)

ds = load_bboxes.from_via_tracks_file(
    file_path, use_frame_numbers_from_file=True
)

# %%
# The loaded dataset is made up of three data arrays:
# ``position``, ``shape``, and ``confidence``.
print(ds)

# %%
# We can see the coordinates in the time dimension are expressed in frames,
# and that we only have data for 1 in 5 frames of the video, plus
# the last frame (167).
#
# In the following sections of the notebook we will explore options to reindex
# the dataset and fill in values for the frames with missing data.
print(ds.time)

# %%
# Inspect associated video
# --------------------------------
# The video associated to the data contains all 168 frames.

video_path = dataset_dict["video"]

video = sio.load_video(video_path)
n_frames, height, width, channels = video.shape

print(f"Number of frames: {n_frames}")
print(f"Frame size: {width}x{height}")
print(f"Number of channels: {channels}")


# %%
# We can plot the data over the corresponding video frames to
# visualise the bounding boxes around the tracked crab.
#
# Let's focus on the first 15 frames of the video, and plot the annotated
# bounding box and centroid at each frame. The centroid at each frame is
# marked as a blue marker with a red ring. The past centroid positions are
# shown in blue and the future centroid positions in white.
#
# Note that in this case the camera is not static relative to the scene.

# select indices of data to plot
data_start_idx = 0
data_end_idx = 15

# initialise figure
fig = plt.figure(figsize=(8, 20))  # width, height

# get list of colors for plotting
list_colors = plt.get_cmap("tab10").colors

# loop over data and plot over corresponding frame
for p_i, data_idx in enumerate(range(data_start_idx, data_end_idx)):
    # add subplot axes
    ax = plt.subplot(math.ceil(data_end_idx / 2), 2, p_i + 1)

    # plot frame
    # note: the video is indexed at every frame, so
    # we use the frame number as index
    ax.imshow(video[ds.time[data_idx].item()])

    # plot box at this frame
    top_left_corner = (
        ds.position[data_idx, 0, :].data - ds.shape[data_idx, 0, :].data / 2
    )
    bbox = plt.Rectangle(
        xy=tuple(top_left_corner),
        width=ds.shape[data_idx, 0, 0].data,  # x coordinate of shape array
        height=ds.shape[data_idx, 0, 1].data,  # y coordinate of shape array
        edgecolor=list_colors[0],
        facecolor="none",
        linewidth=1.5,
    )
    ax.add_patch(bbox)

    # plot box's centroid at this frame with red ring
    ax.scatter(
        x=ds.position[data_idx, 0, 0].data,
        y=ds.position[data_idx, 0, 1].data,
        s=15,
        color=list_colors[0],
        edgecolors="red",
    )

    # plot past centroid positions in blue
    ax.scatter(
        x=ds.position[:data_idx, 0, 0].data,
        y=ds.position[:data_idx, 0, 1].data,
        s=5,
        color=list_colors[0],
    )

    # plot future centroid positionsin white
    ax.scatter(
        x=ds.position[data_idx + 1 : data_end_idx, 0, 0].data,
        y=ds.position[data_idx + 1 : data_end_idx, 0, 1].data,
        s=5,
        color="white",
    )

    ax.set_title(f"Frame {ds.time[data_idx].item()}")
    ax.set_xlabel("x (pixles)")
    ax.set_ylabel("y (pixels)")
    ax.set_xlabel("")

fig.tight_layout()


# %%
# Fill in empty values with forward filling
# ----------------------------------------------------
# We can fill in the frames with missing values for the  ``position`` and
# ``shape`` arrays by taking the last valid value in time. In this way, a
# box's position and shape stay constant if for a current frame the box
# has no annotation defined.

ds_ff = ds.reindex(
    {"time": list(range(ds.time[-1].item()))},
    method="ffill",  # propagate last valid index value forward
)

# check the first 14 frames of the data
print("Position data array (first 14 frames):")
print(ds_ff.position.data[:14, 0, :])  # time, individual, space

print("----")
print("Shape data array (first 14 frames):")
print(ds_ff.shape.data[:14, 0, :])  # time, individual, space

# %%
# Fill in empty values with NaN
# ----------------------------------------------------
# Alternatively, we can fill in the missing frames with NaN values.
# This can be useful if we want to interpolate the missing values later.
ds_nan = ds.reindex(
    {"time": list(range(ds.time[-1].item()))},
    method=None,  # default
)

# check the first 14 frames of the data
print("Position data array (first 14 frames):")
print(ds_nan.position.data[:14, 0, :])

print("----")
print("Shape data array (first 14 frames):")
print(ds_nan.shape.data[:14, 0, :])

# %%
# Linearly interpolate NaN values
# ----------------------------------------------------------
# We can instead fill in the missing values in the dataset applying linear
# interpolation to the ``position`` and ``shape`` data arrays. In this way,
# we would be assuming that the centroid of the bounding box moves linearly
# between the two annotated values, and its width and height change linearly
# as well.
#
# We use the dataset with NaN values as an input to the
# ``interpolate_over_time`` function.
ds_interp = ds_nan.copy()

for data_array_str in ["position", "shape"]:
    ds_interp[data_array_str] = interpolate_over_time(
        data=ds_interp[data_array_str],
        method="linear",
        max_gap=None,
        print_report=False,
    )

# check the first 14 frames of the data
print("Position data array (first 14 frames):")
print(ds_interp.position.data[:14, 0, :])

print("----")
print("Shape data array (first 14 frames):")
print(ds_interp.shape.data[:14, 0, :])


# %%
# Compare interpolation methods
# ------------------------------
# We can now qualitatively compare the three different methods of filling
# in the missing frames, by plotting the bounding boxes
# for the first 6 frames of the video.
#
# Remember only frames 0 and 5 are annotated in the original dataset. These
# are plotted in blue, while the forward filled values are plotted in orange
# and the linearly interpolated values in green.

# sphinx_gallery_thumbnail_number = 2

# initialise figure
fig = plt.figure(figsize=(8, 8))

# loop over frames
for frame_n in range(6):
    # add subplot axes
    ax = plt.subplot(3, 2, frame_n + 1)

    # plot frame
    # note: the video is indexed at every frame, so
    # we use the frame number as index
    ax.imshow(video[frame_n])

    # plot bounding box for each dataset
    for ds_i, ds_one in enumerate([ds_nan, ds_ff, ds_interp]):
        # plot box
        top_left_corner = (
            ds_one.position.sel(time=frame_n, individuals="id_1").data
            - ds_one.shape.sel(time=frame_n, individuals="id_1").data / 2
        )
        bbox = plt.Rectangle(
            xy=tuple(top_left_corner),
            width=ds_one.shape.sel(
                time=frame_n, individuals="id_1", space="x"
            ).data,
            height=ds_one.shape.sel(
                time=frame_n, individuals="id_1", space="y"
            ).data,
            edgecolor=list_colors[ds_i],
            facecolor="none",
            # make line for NaN dataset thicker and dotted
            linewidth=[5, 1.5, 1.5][ds_i],
            linestyle=["dotted", "solid", "solid"][ds_i],
            label=["nan", "ffill", "linear"][ds_i],
        )
        ax.add_patch(bbox)

        # plot centroid
        ax.scatter(
            x=ds_one.position.sel(
                time=frame_n, individuals="id_1", space="x"
            ).data,
            y=ds_one.position.sel(
                time=frame_n, individuals="id_1", space="y"
            ).data,
            s=5,
            color=list_colors[ds_i],
        )

    # add legend to first frame
    if frame_n == 0:
        ax.legend()
    ax.set_title(f"Frame {frame_n}")
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")

fig.tight_layout()

# %%
# Export as .csv file
# -------------------
# Let's assume the dataset with the forward filled values is the best suited
# for our task - we can now export the computed values to a .csv file
#
# Note that we currently do not provide explicit methods to export a
# ``movement`` bounding boxes dataset in a specific format. However, we can
# easily save the bounding boxes’ trajectories to a .csv file using the
# standard Python library ``csv``.

# define name for output csv file
filepath = "tracking_output.csv"

# open the csv file in write mode
with open(filepath, mode="w", newline="") as file:
    writer = csv.writer(file)

    # write the header
    writer.writerow(
        ["frame_idx", "bbox_ID", "x", "y", "width", "height", "confidence"]
    )

    # write the data
    for individual in ds.individuals.data:
        for frame in ds.time.data:
            x, y = ds.position.sel(time=frame, individuals=individual).data
            width, height = ds.shape.sel(
                time=frame, individuals=individual
            ).data
            confidence = ds.confidence.sel(
                time=frame, individuals=individual
            ).data
            writer.writerow(
                [frame, individual, x, y, width, height, confidence]
            )

# %%
# Remove the output file
# ----------------------
# To remove the output file we have just created, we can run the following
# code.
os.remove(filepath)
