"""Load and view FreeMoCap Data
==========================================

Load the ``.csv`` files from FreeMoCap into movement and visualise them.
"""

# %%
# Imports
# -------
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import xarray as xr
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from movement import sample_data
from movement.io import load_poses
from movement.kinematics import compute_speed

# %%
# Load sample dataset
# -------------------
# In this tutorial, we will show how we can import 3D data
# collected with
# `FreeMoCap <https://freemocap.org>`_
# into ``movement``.
#
# FreeMoCap organises the collected data into timestamped `session`
# directories, which hold ``recordings``. For each recording,
# FreeMoCap produces several ``.csv`` files
# (under the ``output_data`` directory),
# each referring to different models
# (e.g. ``face``, ``body``, ``left_hand``, ``right_hand``).
# The output ``.csv`` files will have 3xN columns, with N being the
# number of keypoints used by the model.
#
# Here we will demonstrate how we can load all output files from a single
# recording into a unified ``movement`` dataset.
#
# To do this, we will use a FreeMoCap dataset of a single
# session folder with two recordings. In the first recording,
# a human writes the word "hello" in the air
# with their index finger ("recording_15_37_37_gmt+1").
# In the second recording, they write the word "world"
# ("recording_15_45_49_gmt+1"). Let's first fetch the
# dataset using the ``sample_data`` module.
session_dir_path = Path(
    sample_data.fetch_dataset_paths(
        "FreeMoCap_hello-world_session-folder.zip"
    )["poses"]
)

print("Path to recordings: ", session_dir_path)

recording_dir_hello = session_dir_path.joinpath(
    "recording_15_37_37_gmt+1/output_data"
)
recording_dir_world = session_dir_path.joinpath(
    "recording_15_45_49_gmt+1/output_data"
)


# %%
# Load FreeMoCap output files as a single ``movement`` dataset
# ---------------------------------
# We will use the helper function below to combine all FreeMoCap output
# ``.csv`` files into one ``movement`` dataset.
def read_freemocap_as_ds(recording_dir_path, individual_name="person_0"):
    model = ["body", "left_hand", "right_hand", "face"]
    full_data = xr.Dataset()
    for c in model:
        data_pd = pd.read_csv(
            recording_dir_path.joinpath(f"mediapipe_{c}_3d_xyz.csv")
        )
        headers = data_pd.columns
        data = data_pd.to_numpy()
        data = np.array(
            [data.reshape(data.shape[0], int(data.shape[1] / 3), 3)]
        )
        # Transpose to align with movement data shape
        data_transposed = np.transpose(data, [1, 3, 2, 0])
        ds = load_poses.from_numpy(
            position_array=data_transposed,
            confidence_array=np.ones(np.delete(data_transposed.shape, 1)),
            individual_names=[individual_name],
            # Select and trim header names
            keypoint_names=[a[:-2] for a in headers[::3]],
        )
        # Merge all of the Datasets into one, along the keypoints axis
        full_data = full_data.merge(ds, 2)
    return full_data


# We can now use the helper function to read the files
# in each recording directory as a ``movement`` dataset

ds_hello = read_freemocap_as_ds(recording_dir_hello)
ds_world = read_freemocap_as_ds(recording_dir_world)

# Note that each ``movement`` dataset holds the data for
# all keypoints across all models used by FreeMoCap.

# %%
# Visualising the data
# --------------------
# Selecting and adjusting the data before plotting.

# We trim the data to select a specific time window
# (when the desired movement is taking place)
# and selecting the individual.
ds_hello = ds_hello.sel(time=range(30, 180), individuals="person_0")
ds_world = ds_world.sel(time=range(150), individuals="person_0")

# Select the ``keypoint`` ``right_hand_0006`` (in the finger),
# which is being used for writing.
position_hello = ds_hello.position.sel(keypoints="right_hand_0006")
position_world = ds_world.position.sel(keypoints="right_hand_0006")


# %%
# 3D coloured line function
# This function acts to render a multi-coloured line. Adapted from
# `Matplotlib segmented example <https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html>`_.
def coloured_scatter_line_3d(x, y, z, c, s, ax, **lc_kwargs):
    ax.scatter(x, y, z, c=c, s=s, **lc_kwargs)

    x, y, z = (np.asarray(arr).ravel() for arr in (x, y, z))
    default_kwargs = {"capstyle": "butt"}
    default_kwargs.update(lc_kwargs)

    x_mid = np.hstack((x[:1], 0.5 * (x[1:] + x[:-1]), x[-1:]))
    y_mid = np.hstack((y[:1], 0.5 * (y[1:] + y[:-1]), y[-1:]))
    z_mid = np.hstack((z[:1], 0.5 * (z[1:] + z[:-1]), z[-1:]))

    start = np.column_stack((x_mid[:-1], y_mid[:-1], z_mid[:-1]))[:, None, :]
    mid = np.column_stack((x, y, z))[:, None, :]
    end = np.column_stack((x_mid[1:], y_mid[1:], z_mid[1:]))[:, None, :]

    segments = np.concatenate((start, mid, end), axis=1)

    lc = Line3DCollection(segments, **default_kwargs)
    lc.set_array(c)
    ax.add_collection3d(lc)

    return lc


# %%
# Visualising the skeleton
# ------------------------
# Using
# `NetworkX <https://networkx.org/>`_
# to join some keypoints and construct a skeleton
# of the upper-half of the individual.

# Select some keypoints from the `body` model
selected_body_kpts = [
    "body_nose",
    "body_left_eye",
    "body_right_eye",
    "body_left_shoulder",
    "body_right_shoulder",
    "body_left_hip",
    "body_right_hip",
    "body_left_elbow",
    "body_right_elbow",
    "body_left_wrist",
    "body_right_wrist",
    "body_left_index",
    "body_right_index",
    "body_left_pinky",
    "body_right_pinky",
    "body_left_thumb",
    "body_right_thumb",
    "body_left_ear",
    "body_right_ear",
    "body_mouth_left",
    "body_mouth_right",
]

# Choose a frame index to create the body from
body_frame = 130

# Initialize graph
G = nx.Graph()

# Add nodes
for kpt in selected_body_kpts:
    G.add_node(
        kpt,
        position=ds_hello.position.sel(
            time=body_frame, keypoints=kpt
        ).values,  # (n_frames, 3)
    )

# Add midpoint between the shoulders as node
G.add_node(
    "body_shoulders_midpoint",
    position=np.mean(
        ds_hello.position.sel(
            time=body_frame,
            keypoints=["body_left_shoulder", "body_right_shoulder"],
        ).values,
        axis=1,
    ),
)

# Add edges
G.add_edge("body_right_shoulder", "body_left_shoulder")
for side_str in ["left", "right"]:
    G.add_edge(f"body_{side_str}_shoulder", f"body_{side_str}_elbow")
    G.add_edge(f"body_{side_str}_shoulder", f"body_{side_str}_hip")
    G.add_edge(f"body_{side_str}_elbow", f"body_{side_str}_wrist")
    G.add_edge(f"body_{side_str}_wrist", f"body_{side_str}_index")
    G.add_edge(f"body_{side_str}_wrist", f"body_{side_str}_pinky")
    G.add_edge(f"body_{side_str}_wrist", f"body_{side_str}_thumb")
    G.add_edge(f"body_mouth_{side_str}", f"body_{side_str}_ear")

# Add edge between the shoulders midpoint and the nose
G.add_edge("body_shoulders_midpoint", "body_nose")

# Add edge between the ears
G.add_edge("body_left_ear", "body_right_ear")

# Add edge across the mouth
G.add_edge("body_mouth_left", "body_mouth_right")

fig_a = plt.figure()
ax_a = fig_a.add_subplot(111, projection="3d")

# Get positions dictionary (key: node, value: (x, y, z))
positions_dict = nx.get_node_attributes(G, "position")

# Plot nodes
for _node, (x, y, z) in positions_dict.items():
    ax_a.scatter(x, y, z, s=10, color="green")

# Plot edges
for edge in G.edges():
    node_1, node_2 = edge
    x1, y1, z1 = positions_dict[node_1]
    x2, y2, z2 = positions_dict[node_2]
    ax_a.plot([x1, x2], [y1, y2], [z1, z2], "b-")

# Plot the text keypoints for all frames
ax_a.plot(
    position_hello.sel(space="x"),
    position_hello.sel(space="y"),
    position_hello.sel(space="z"),
    alpha=0.35,
    color="magenta",
)

ax_a.view_init(elev=25, azim=-120, roll=0)
ax_a.set_aspect("equal")
ax_a.set_xlabel("x (mm, inverted)")
ax_a.set_ylabel("y (mm)")
ax_a.set_zlabel("z (mm)")

# Invert ``x-axis`` to make traced text readable
ax_a.invert_xaxis()


# %%
# Displaying speed and time as colours
# ------------------------------------
# Visualising the frame-index
# ``x, y, z`` where time determines colour.
fig_b = plt.figure()
ax_b = fig_b.add_subplot(projection="3d")

colour_map_b = "turbo"
# Use time component of Dataset
frame_hello = ds_hello.time
frame_world = ds_world.time

# Add "hello" scatter and line

coloured_scatter_line_3d(
    position_hello.sel(space="x"),
    position_hello.sel(space="y"),
    position_hello.sel(space="z"),
    ax=ax_b,
    c=frame_hello,
    s=5,
    cmap=colour_map_b,
)
# Add "world" scatter and line

coloured_scatter_line_3d(
    position_world.sel(space="x"),
    position_world.sel(space="y"),
    position_world.sel(space="z") - 400,
    ax=ax_b,
    c=frame_world,
    s=5,
    cmap=colour_map_b,
)

# Change view orientation
ax_b.view_init(elev=-20, azim=137, roll=0)
ax_b.set_aspect("equal")
ax_b.set_xlabel("x (mm)")
ax_b.set_ylabel("y (mm)")
ax_b.set_zlabel("z (mm)")
ax_b.set_title("Visualising frame-index", pad=40)

# Next, visualising the current writing speed
# ``x, y, z`` where speed determines colour.
fig_c = plt.figure()
ax_c = fig_c.add_subplot(projection="3d")

colour_map_c = "inferno"

# Use movement helpers to compute speed
speed_hello = compute_speed(position_hello)
speed_world = compute_speed(position_world)

# Add "hello" scatter and line
coloured_scatter_line_3d(
    position_hello.sel(space="x"),
    position_hello.sel(space="y"),
    position_hello.sel(space="z"),
    ax=ax_c,
    c=speed_hello,
    s=5,
    cmap=colour_map_c,
)
# Add "world" scatter and line
coloured_scatter_line_3d(
    position_world.sel(space="x"),
    position_world.sel(space="y"),
    position_world.sel(space="z") - 400,
    ax=ax_c,
    c=speed_world,
    s=5,
    cmap=colour_map_c,
)

# Change view orientation
ax_c.view_init(elev=-20, azim=137, roll=0)
ax_c.set_aspect("equal")
ax_c.set_xlabel("x (mm)")
ax_c.set_ylabel("y (mm)")
ax_c.set_zlabel("z (mm)")
ax_c.set_title("Visualising writing speed", pad=40)
# %%
