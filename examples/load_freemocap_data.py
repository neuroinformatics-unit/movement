"""Load and visualise FreeMoCap data
==========================================

Load the ``.csv`` files from FreeMoCap into ``movement`` and visualise them.
"""

# %%
# Imports
# -------

# To run this example, you will need to install the ``networkx`` package.
# You can do this by running ``pip install networkx`` in your active
# virtual environment.

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
# Download sample dataset
# -----------------------
# In this tutorial, we will show how we can import 3D data
# collected with
# `FreeMoCap <https://freemocap.org>`_
# into ``movement``.
#
# FreeMoCap organises the collected data into timestamped ``session``
# directories, each of which hold ``recording`` subdirectories.
# For each recording, FreeMoCap saves several ``.csv`` files
# under the ``output_data`` directory. Each ``.csv`` file is
# named after the model used to produce the data (e.g. ``face``,
# ``body``, ``left_hand``, ``right_hand``).
# The output ``.csv`` files will have 3N columns, with N being the
# number of keypoints used by the relevant model and 3 being the number of
# spatial dimensions (x, y, z).
#
# Here we will demonstrate how we can load all output files from a single
# recording into a unified ``movement`` dataset.
#
# To do this, we will use a FreeMoCap dataset of a single
# session folder with two recordings. In the first recording,
# a human writes the word "hello" in the air
# with their index finger (``recording_15_37_37_gmt+1``).
# In the second recording, they write the word "world"
# (``recording_15_45_49_gmt+1``). Let's first fetch the
# dataset using the ``sample_data`` module.
session_dir_path = sample_data.fetch_dataset_paths(
    "FreeMoCap_hello-world_session-folder.zip"
)["poses"]

# path to output data directory for "hello" recording
output_data_dir_hello = session_dir_path.joinpath(
    "recording_15_37_37_gmt+1/output_data"
)

# path to output data directory for "world" recording
output_data_dir_world = session_dir_path.joinpath(
    "recording_15_45_49_gmt+1/output_data"
)

print("Path to session folder: ", session_dir_path)
print(
    "Path to output data directory for 'hello' recording: ",
    output_data_dir_hello,
)
print(
    "Path to output data directory for 'world' recording: ",
    output_data_dir_world,
)


# %%
# Load FreeMoCap output files as a single ``movement`` dataset
# ------------------------------------------------------------
# We will use the helper function below to combine all FreeMoCap output
# ``.csv`` files into one ``movement`` dataset. Since there is no confidence
# data available in the output files, we will set the confidence of all
# keypoints to the default NaN value.


def read_freemocap_as_ds(output_data_dir, individual_name="id_0"):
    """Read FreeMoCap output files as a single ``movement`` dataset.

    Parameters
    ----------
    output_data_dir : pathlib.Path
        Path to the recording's output data directory holding the relevant
        ``.csv`` files.
    individual_name : str, optional
        Name of the individual to be used as the ``individual`` dimension
        in the returned ``movement`` dataset. Defaults to ``id_0``.

    Returns
    -------
    xarray.Dataset
        A ``movement`` dataset containing the data from all FreeMoCap output
        files. The ``keypoints`` dimension will have the full set of keypoints
        as coordinates.

    """
    models = ["body", "left_hand", "right_hand", "face"]
    # list_datasets = []
    ds_all_keypoints = xr.Dataset()
    for m in models:
        # Read .csv file as a pandas DataFrame
        data_pd = pd.read_csv(
            output_data_dir.joinpath(f"mediapipe_{m}_3d_xyz.csv")
        )

        # Get headers from DataFrame
        headers = data_pd.columns
        keypoint_names = list(set(header[:-2] for header in headers))

        # Format data as numpy array
        data = data_pd.to_numpy()
        data = data.reshape(
            data.shape[0], int(data.shape[1] / 3), 3
        )  # time, keypoint, space

        # Transpose to align with movement dimensions
        # and add singleton individuals dimension at the end
        data = np.transpose(data, [0, 2, 1])[..., None]

        # Read as a movement dataset
        ds = load_poses.from_numpy(
            position_array=data,  # time, space, keypoint, individual
            individual_names=[individual_name],
            keypoint_names=keypoint_names,
        )
        ds_all_keypoints = ds_all_keypoints.merge(ds, 2)

    # Merge all datasets along keypoint dimension
    # ds_all_keypoints = xr.merge(list_datasets)
    return ds_all_keypoints


# %%
# We can now use the helper function to read the files
# in each recording directory as a ``movement`` dataset.

ds_hello = read_freemocap_as_ds(output_data_dir_hello)
ds_world = read_freemocap_as_ds(output_data_dir_world)

# %%
# Note that each ``movement`` dataset holds the data for
# all keypoints across all models used by FreeMoCap.

print(f"Number of keypoints in 'hello' dataset: {len(ds_hello.keypoints)}")
print(f"Number of keypoints in 'world' dataset: {len(ds_world.keypoints)}")


# %%
# Prepare the data for visualisation
# ----------------------------------
# To better visualise the data, we will need to wrangle the data a bit.
# We will also focus on the ``position`` data array of each dataset.

# First, we will focus on the time window when the desired movement
# takes place.
position_hello = ds_hello.position.sel(time=range(30, 180))
position_world = ds_world.position.sel(time=range(150))


# %%
# Next, we will use the ``networkx`` package to define a graph based on a
# subset of keypoints. This will allow us to construct a skeleton of the
# upper-half of the individual.

# %%
# Select some keypoints from the ``body`` model.
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

# Add nodes to the graph.
for kpt in selected_body_kpts:
    G.add_node(
        kpt,
        position=ds_hello.position.sel(
            time=body_frame,
            keypoints=kpt,  # , individuals="id_0"
        ).values,  # (n_frames, 3)
    )

# Add midpoint between the shoulders as node
G.add_node(
    "body_shoulders_midpoint",
    position=np.mean(
        ds_hello.position.sel(
            time=body_frame,
            keypoints=["body_left_shoulder", "body_right_shoulder"],
            # individuals="id_0",
        ).values,
        axis=1,
    ),
)

# Add edges to the graph.
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


# %%
# Visualise the skeleton
# ---------------------
# We can now visualise the skeleton.

# %%
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

# Plot the text keypoints for the selected time window
ax_a.plot(
    position_hello.sel(
        space="x", keypoints="right_hand_0006", individuals="id_0"
    ),
    position_hello.sel(
        space="y", keypoints="right_hand_0006", individuals="id_0"
    ),
    position_hello.sel(
        space="z", keypoints="right_hand_0006", individuals="id_0"
    ),
    alpha=0.35,
    color="magenta",
)

ax_a.view_init(elev=25, azim=-120, roll=0)
ax_a.set_aspect("equal")
ax_a.set_xlabel("x (mm, inverted)")
ax_a.set_ylabel("y (mm)")
ax_a.set_zlabel("z (mm)")
ax_a.set_title("Visualising the skeleton", pad=10)

# Invert ``x-axis`` to make traced text readable
ax_a.invert_xaxis()

# %%
# We will also use the following helper function to visualise the data.
# This function is adapted from
# ``matplotlib``'s multi-coloured line example
# <https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html>_.
# You don't need to understand how this function works in detail, for now it
# is enough to know that we will use it to plot 3D lines with a colour
# determined by a scalar.


def coloured_scatter_line_3d(x, y, z, c, s, ax, **lc_kwargs):
    """Plot a 3D line and colour by a scalar.

    Parameters
    ----------
    x, y, z : array-like
        x, y, z-coordinates of the line to plot.
    c : array-like
        Scalar valuesto colour the line by.
    s : float
        Size of the markers on the line.
    ax : matplotlib.axes.Axes
        Axes to plot the line on.
    **lc_kwargs : dict
        Keyword arguments to pass to ``Line3DCollection``.

    Returns
    -------
    lc : matplotlib.collections.Line3DCollection
        The line collection.

    """
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
# Displaying time and speed as colours
# ------------------------------------
# Visualising the frame-index
# ``x, y, z`` where time determines colour.
# The line colour progresses through the ``turbo`` colormap as the
# individual writes more letters.
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
cb_b = fig_b.colorbar(
    ax_b.collections[0],
    ax=ax_b,
    label="Frame Index",
    fraction=0.06,
    orientation="horizontal",
    pad=-0.05,
)

# %%
# Next, visualising the current writing speed
# ``x, y, z`` where speed determines colour.
# Using :func:`compute_speed()\
# <movement.kinematics.compute_speed>` to compute the speed
# of the writing movement, and mapping the line colour to the
# ``inferno`` colormap.
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
cb_c = fig_c.colorbar(
    ax_c.collections[0],
    ax=ax_c,
    label="Speed (mm/s)",
    fraction=0.06,
    orientation="horizontal",
    pad=-0.05,
)

# We can see that the speed is highest on long, straight
# strokes and lowest around kinks.
# %%
