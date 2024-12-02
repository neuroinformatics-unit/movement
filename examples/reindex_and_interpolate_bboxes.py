"""Reindex and interpolate bounding boxes tracks
===============================

Load an example dataset of bounding boxes' tracks and reindex
it to every frame.
"""

# %%
import math

import sleap_io as sio
from cycler import cycler
from matplotlib import pyplot as plt

from movement import sample_data
from movement.filtering import interpolate_over_time
from movement.io import load_bboxes

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Select sample data file
# --------------------
# For this example, we will use the path to one of
# the sample datasets provided with ``movement``.

dataset_dict = sample_data.fetch_dataset_paths(
    "VIA_single-crab_MOCA-crab-1.csv",
    with_video=True,  # for visualisation
)

file_path = dataset_dict["bboxes"]
print(file_path)

ds = load_bboxes.from_via_tracks_file(
    file_path, use_frame_numbers_from_file=True
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Only 1 in 5 frames are annotated, plus the last frame (167)
print(ds)
print("-----")
print(ds.time)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Extend the dataset to every frame by forward filling
# The position and shape data arrays are filled with the last valid value
# So position and shape are kept constant when no annotation is available
ds_ff = ds.reindex(
    {"time": list(range(ds.time[-1].item()))},
    method="ffill",  # propagate last valid index value forward
)

print("Position data array (first 14 frames):")
print(ds_ff.position.data[:14, 0, :])  # time, individual, space

print("----")
print("Shape data array (first 14 frames):")
print(ds_ff.shape.data[:14, 0, :])  # time, individual, space

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Extend the dataset to every frame and fill empty values with nan
ds_nan = ds.reindex(
    {"time": list(range(ds.time[-1].item()))},
    method=None,  # default
)

print("Position data array (first 14 frames):")
print(ds_nan.position.data[:14, 0, :])

print("----")
print("Shape data array (first 14 frames):")
print(ds_nan.shape.data[:14, 0, :])

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Linearly interpolate position and shape with nan

ds_interp = ds_nan.copy()

for data_array_str in ["position", "shape"]:
    ds_interp[data_array_str] = interpolate_over_time(
        data=ds_interp[data_array_str],
        method="linear",
        max_gap=None,
        print_report=False,
    )

print("Position data array (first 14 frames):")
print(ds_interp.position.data[:14, 0, :])

print("----")
print("Shape data array (first 14 frames):")
print(ds_interp.shape.data[:14, 0, :])


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Inspect associated video

video_path = dataset_dict["video"]


video = sio.load_video(video_path)

n_frames, height, width, channels = video.shape

print(f"Number of frames: {n_frames}")  # The video contains all frames
print(f"Frame size: {width}x{height}")
print(f"Number of channels: {channels}")


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Plot data
# OJO camera movement

# select indices of data to plot
data_start_idx = 0
data_end_idx = 11

# initialise figure
fig = plt.figure(figsize=(15, 12))

# add color cycler to axes
plt.rcParams["axes.prop_cycle"] = cycler(color=plt.get_cmap("tab10").colors)
color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# loop over data and plot over corresponding frame
for p_i, data_idx in enumerate(range(data_start_idx, data_end_idx)):
    # add subplot axes
    ax = plt.subplot(math.ceil(data_end_idx / 5), 5, p_i + 1)

    # plot frame
    ax.imshow(
        video[ds.time[data_idx].item()]
    )  # the video is indexed at every frame! use frame number as index

    # plot annotated boxes
    top_left_corner = (
        ds.position[data_idx, 0, :].data - ds.shape[data_idx, 0, :].data / 2
    )
    bbox = plt.Rectangle(
        xy=tuple(top_left_corner),
        width=ds.shape[data_idx, 0, 0].data,  # x coord
        height=ds.shape[data_idx, 0, 1].data,  # y coord of shape array
        edgecolor=color_cycle[0],  # [data_idx % len(color_cycle)],
        facecolor="none",  # transparent fill
        linewidth=1.5,
    )
    ax.add_patch(bbox)

    ax.set_title(f"Frame {ds.time[data_idx].item()}")

fig.tight_layout()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Compare interpolation methods

# select frames to inspect
frame_number_start = 0
frame_number_end = 6

# add color cycler to axes
plt.rcParams["axes.prop_cycle"] = cycler(color=plt.get_cmap("tab10").colors)
color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]


# initialise figure
fig = plt.figure(figsize=(15, 12))


# loop over data and plot over corresponding frame
for frame_n in range(frame_number_start, frame_number_end):
    # add subplot axes
    ax = plt.subplot(1, 6, frame_n + 1)

    # plot frame
    ax.imshow(video[frame_n])
    # the video is indexed at every frame! use frame number as index

    # plot bounding box: box and centroid
    for ds_i, ds in enumerate([ds_nan, ds_ff, ds_interp]):
        # plot box
        top_left_corner = (
            ds.position.sel(time=frame_n, individuals="id_1").data
            - ds.shape.sel(time=frame_n, individuals="id_1").data / 2
        )
        bbox = plt.Rectangle(
            xy=tuple(top_left_corner),
            width=ds.shape.sel(
                time=frame_n, individuals="id_1", space="x"
            ).data,  # x coord
            height=ds.shape.sel(
                time=frame_n, individuals="id_1", space="y"
            ).data,  # y coord of shape array
            edgecolor=color_cycle[ds_i],
            facecolor="none",  # transparent fill
            linewidth=[4.5, 1.5, 1.5][ds_i],
            linestyle=["dotted", "solid", "solid"][ds_i],
            label=["nan", "ffill", "linear"][ds_i],
        )
        ax.add_patch(bbox)

        # plot centroid
        ax.scatter(
            x=ds.position.sel(
                time=frame_n, individuals="id_1", space="x"
            ).data,
            y=ds.position.sel(
                time=frame_n, individuals="id_1", space="y"
            ).data,
            s=5,
            color=color_cycle[ds_i],
        )

    if frame_n == 0:
        ax.legend()
    ax.set_title(f"Frame {frame_n}")

fig.tight_layout()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Export as csv file
