"""Load FreeMoCap Data
==========================================

Load the ..
"""

# %%
import os as os

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from movement.io import load_poses
from movement.kinematics import compute_speed


# %%
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

path_hello = (
    "C:/Users/Max/Downloads/session_2025-07-08_15_35_19/"
    "session_2025-07-08_15_35_19/recording_15_37_37_gmt+1/output_data/"
    "mediapipe_right_hand_3d_xyz.npy"
)
path_world = (
    "C:/Users/Max/Downloads/session_2025-07-08_15_35_19/"
    "session_2025-07-08_15_35_19/recording_15_45_49_gmt+1/output_data/"
    "mediapipe_right_hand_3d_xyz.npy"
)
# %%
data_hello = np.load(path_hello)
data_hello = np.array([data_hello])
data_world = np.load(path_world)
data_world = np.array([data_world])
print(data_hello.shape)
# %%
data_transposed_hello = np.transpose(data_hello, [1, 3, 2, 0])
data_transposed_world = np.transpose(data_world, [1, 3, 2, 0])
print(data_transposed_hello.shape)
# %%
ds_hello = load_poses.from_numpy(
    position_array=data_transposed_hello,
    confidence_array=np.ones(np.delete(data_transposed_hello.shape, 1)),
    individual_names=["person_0"],
)
ds_world = load_poses.from_numpy(
    position_array=data_transposed_world,
    confidence_array=np.ones(np.delete(data_transposed_world.shape, 1)),
    individual_names=["person_0"],
)

# %%
ds_hello = ds_hello.sel(time=range(30, 190), individuals="person_0")
ds_world = ds_world.sel(time=range(150), individuals="person_0")

# %%
position_hello = ds_hello.position.sel(keypoints="keypoint_9")
position_world = ds_world.position.sel(keypoints="keypoint_8")


fig3d = plt.figure()
ax3d = fig3d.add_subplot(projection="3d")
x_hello = position_hello.sel(space="x")
z_hello = position_hello.sel(space="z")
y_hello = position_hello.sel(space="y")
x_world = position_world.sel(space="x")
z_world = position_world.sel(space="z")
y_world = position_world.sel(space="y")

speed_hello = compute_speed(position_hello)
speed_world = compute_speed(position_world)

colour_map = "berlin"

ax3d.scatter(x_hello, y_hello, z_hello, c=speed_hello, s=5, cmap=colour_map)
colored_line_3d(
    x_hello, y_hello, z_hello, ax=ax3d, c=speed_hello, cmap=colour_map
)
ax3d.scatter(
    x_world, y_world, z_world - 500, c=speed_world, s=5, cmap=colour_map
)
colored_line_3d(
    x_world, y_world, z_world - 500, ax=ax3d, c=speed_world, cmap=colour_map
)

ax3d.view_init(elev=-20, azim=137, roll=0)
# %%
