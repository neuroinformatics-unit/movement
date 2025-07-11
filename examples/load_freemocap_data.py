"""Load and view FreeMoCap Data
==========================================

Load the ``.csv`` files from FreeMoCap into movement and visualise them.
"""

# %%
# Imports
# -------
import os

import matplotlib.pyplot as plt
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
# We will use two FreeMoCap 3D tracked videos of a person tracing
# out "hello" and "world" with their finger.
session_dir_path = sample_data.fetch_dataset_paths(
    "FreeMoCap_hello-world_session-folder.zip"
)["poses"]

path_hello = os.path.join(
    session_dir_path,
    "recording_15_37_37_gmt+1/output_data",
)
path_world = os.path.join(
    session_dir_path,
    "recording_15_45_49_gmt+1/output_data",
)


# %%
# Load FreeMoCap data into movement
# ---------------------------------
# This function combines all FreeMoCap output
# ``.csv`` files into one ``xarray Dataset``.
def load_body_pose_FMC(folder_path, person_name="person_0"):
    components = ["body", "left_hand", "right_hand", "face"]
    full_data = xr.Dataset()
    for c in components:
        data_pd = pd.read_csv(
            os.path.join(folder_path, f"mediapipe_{c}_3d_xyz.csv")
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
            individual_names=[person_name],
            # Select and trim header names
            keypoint_names=[a[:-2] for a in headers[::3]],
        )
        # Merge all of the Datasets into one, along the keypoints axis
        full_data = full_data.merge(ds, 2)
    return full_data


ds_hello = load_body_pose_FMC(path_hello)
ds_world = load_body_pose_FMC(path_world)
# %%
# Selecting and adjusting the data before plotting.

# Trimming to the correct frames and selecting the individual
ds_hello = ds_hello.sel(time=range(30, 180), individuals="person_0")
ds_world = ds_world.sel(time=range(150), individuals="person_0")

# Select the ``right_hand_0007`` (in the finger)
position_hello = ds_hello.position.sel(keypoints="right_hand_0007")
position_world = ds_world.position.sel(keypoints="right_hand_0007")

x_hello = position_hello.sel(space="x")
z_hello = position_hello.sel(space="z")
y_hello = position_hello.sel(space="y")
x_world = position_world.sel(space="x")
# Separate "world" by translating on z axis
z_world = position_world.sel(space="z") - 400
y_world = position_world.sel(space="y")


# %%
# 3D coloured line function
# -------------------------
# This function acts to render a multi-coloured line. Adapted from
# `Matplotlib segmented example <https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html>`_.
def colored_line_3d(x, y, z, c, ax, **lc_kwargs):
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
# Plot A: Frame
# -------------
# ``x, y, z`` where time determines colour.
fig_a = plt.figure()
axes_a = fig_a.add_subplot(projection="3d")

colour_map_a = "turbo"

# Use time component of Dataset
frame_hello = ds_hello.time
frame_world = ds_world.time

# Add "hello" scatter and line
axes_a.scatter(
    x_hello, y_hello, z_hello, c=frame_hello, s=5, cmap=colour_map_a
)
colored_line_3d(
    x_hello, y_hello, z_hello, ax=axes_a, c=frame_hello, cmap=colour_map_a
)
# Add "world" scatter and line
axes_a.scatter(
    x_world, y_world, z_world, c=frame_world, s=5, cmap=colour_map_a
)
colored_line_3d(
    x_world, y_world, z_world, ax=axes_a, c=frame_world, cmap=colour_map_a
)

# Change view orientation
axes_a.view_init(elev=-20, azim=137, roll=0)
# %%
# Plot B: Speed
# -------------
# ``x, y, z`` where speed determines colour.
fig_b = plt.figure()
axes_b = fig_b.add_subplot(projection="3d")

colour_map_b = "inferno"

# Use movement helpers to compute speed
speed_hello = compute_speed(position_hello)
speed_world = compute_speed(position_world)

# Add "hello" scatter and line
axes_b.scatter(
    x_hello, y_hello, z_hello, c=speed_hello, s=5, cmap=colour_map_b
)
colored_line_3d(
    x_hello, y_hello, z_hello, ax=axes_b, c=speed_hello, cmap=colour_map_b
)
# Add "world" scatter and line
axes_b.scatter(
    x_world, y_world, z_world, c=speed_world, s=5, cmap=colour_map_b
)
colored_line_3d(
    x_world, y_world, z_world, ax=axes_b, c=speed_world, cmap=colour_map_b
)

# Change view orientation
axes_b.view_init(elev=-20, azim=137, roll=0)
