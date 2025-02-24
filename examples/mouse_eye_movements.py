"""Pupillometry
===================================

Look at eye movements caused by the oculo-motor reflex.
"""

# %%
# Imports
# -------
import numpy as np
import sleap_io as sio
import xarray as xr
from matplotlib import pyplot as plt

import movement.kinematics as kin
from movement import sample_data
from movement.plots import plot_trajectory

# %%
# Load the data
# -------------
ds_black = sample_data.fetch_dataset(
    "DLC_rotating-mouse_eye-tracking_stim-black.predictions.h5",
    with_video=True,
)
ds_uniform = sample_data.fetch_dataset(
    "DLC_rotating-mouse_eye-tracking_stim-uniform.predictions.h5",
    with_video=True,
)
# Save data in a dictionary
ds_dict = {"black": ds_black, "uniform": ds_uniform}

# %%
# Explore the data
# ----------------
for ds_name, ds in ds_dict.items():
    video = sio.load_video(ds.video_path)
    # to avoid having to reload the video again we add it to ds as attribute
    ds_dict[ds_name] = ds.assign_attrs({"video": video})
    n_frames, height, width, channels = video.shape
    print(f"Dataset: {ds_name}")
    print(f"Number of frames: {n_frames}")
    print(f"Frame size: {width}x{height}")
    print(f"Number of channels: {channels}\n")

# Both datasets contain videos of a mouse eye under two different conditions.
# %%
# Plot first frame with keypoints
# -------------------------------
fig, ax = plt.subplots(1, 2, figsize=(7.5, 4))
for i, (da_name, ds) in enumerate(ds_dict.items()):
    ax[i].imshow(ds.video[0], cmap="gray")  # plot first video frame
    for keypoint in ds.position.keypoints.values:
        x = ds.position.sel(time=0, space="x", keypoints=keypoint)
        y = ds.position.sel(time=0, space="y", keypoints=keypoint)
        ax[i].scatter(x, y, label=keypoint)  # plot keypoints
    ax[i].legend()
    ax[i].set_title(f"{da_name} (First Frame)")
    ax[i].invert_yaxis()  # because the dataset was collected flipped
plt.tight_layout()
plt.show()

# %%
# Pupil trajectory
# ----------------
# A quick plot of the trajectory of the centre of the pupil
# using ``plot_trajectory`` function in ``movement.plots``.
time_points = ds_black.time[slice(50, 1000)]
position_black = ds_black.position.sel(time=time_points)  # data array to plot
fig, ax = plot_trajectory(position_black, keypoints=["pupil-L", "pupil-R"])
fig.show()

# %%
# Pupil trajectories on top of video frame
# ----------------------------------------
# We can look at pupil trajectories plotted on top of a video frame.
fig, ax = plt.subplots(1, 2, figsize=(11, 3))
for i, (ds_name, ds) in enumerate(ds_dict.items()):
    ax[i].imshow(ds.video[100], cmap="gray")  # Plot frame 100 as background

    plot_trajectory(
        ds.position.sel(time=ds.time[slice(50, 1000)]),  # Select time window
        ax=ax[i],
        keypoints=["pupil-L", "pupil-R"],
        alpha=0.5,
        s=3,
    )

    ax[i].invert_yaxis()
    ax[i].set_title(f"Pupil Trajectory ({ds_name})")
plt.show()


# %%
# Keypoint positions over time
# ----------------------------
# For the rest of this example we are interested in the position data only, to
# make the data easy to access we save it in a ``dict`` with the dataset names
# as keys.
position_dict = {"black": ds_black.position, "uniform": ds_uniform.position}


# %%
# We create a function to plot the x and y positions of the keypoints over
# time, so that we can compare the data before and after normalisation.
def plot_x_y_keypoints(da_dict, ylim=None, **kwargs):
    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    for i, ds_name in enumerate(list(da_dict.keys())):
        for j, coord in enumerate(["x", "y"]):
            # Prepare a data array for plotting (squeeze removes redundant
            # dimensions and is needed in order to plot the data correctly)
            da = da_dict[ds_name].sel(space=coord, **kwargs).squeeze()
            da.plot.line(ax=axes[i, j], x="time")  # Plot data
            axes[i, j].set_title(f"{ds_name} - {coord} position")
    if ylim is not None:
        for ax in axes.flat:
            ax.set_ylim(ylim)
    plt.tight_layout()
    return fig


# %%
# Using the function created above, we plot the x and y positions in a
# specified time window (frames 350 to 1000) of the unprocessed data.
time_points = position_dict["black"].time[slice(350, 1001)]
fig = plot_x_y_keypoints(position_dict, ylim=(100, 500), time=time_points)
fig.show()

# %%
# Normalised keypoint positions over time
# ---------------------------------------
# Normalizing the pupil's position relative to the midpoint of the eye reduces
# the impact of head movements or artefacts caused by movement of the camera.
# In the rest of the example, the normalised data will be used.
position_norm_dict = {}
for da_name, da in position_dict.items():
    eye_midpoint = da.sel(keypoints=["eye-L", "eye-R"]).mean("keypoints")
    position_norm_dict[da_name] = da - eye_midpoint
# %%
# We plot the x and y positions again, but now using the processed data.
fig = plot_x_y_keypoints(
    position_norm_dict, ylim=(-200, 200), time=time_points
)
fig.show()
# %%
# Pupil position over time
# ---------------------------------
# A pupil centroid keypoint can be added to the data array using
# ``xarray.DataArray.assign_coords`` and ``xarray.concat``.
for da_name, da in position_norm_dict.items():
    pupil_centroid = da.sel(keypoints=["pupil-L", "pupil-R"]).mean("keypoints")
    pupil_centroid = pupil_centroid.assign_coords({"keypoints": "pupil-C"})
    position_norm_dict[da_name] = xr.concat([da, pupil_centroid], "keypoints")

# %%
# Now the position of the pupil centroid ("pupil-C") can be plotted.
fig, axes = plt.subplots(2, 1, figsize=(6, 4))
for i, (da_name, da) in enumerate(position_norm_dict.items()):
    da = da.sel(keypoints="pupil-C", time=time_points)  # select data to plot
    for coord, color in ["x", "C0"], ["y", "C1"]:
        axes[i].plot(
            time_points,
            da.sel(space=coord),
            label=coord,
            color=color,
        )
    axes[i].set_xlabel("Time (s)")
    axes[i].set_ylabel("Position")
    axes[i].set_title(f"Pupil centroid position ({da_name})")
    axes[i].legend()

for ax in axes.flat:
    ax.set_ylim(-30, 65)

plt.tight_layout()
plt.show()
# %%
# Pupil velocity over time
# ------------------------
# We use ``compute_velocity`` from ``movement``'s ``kinematics`` module to
# calculate the velocity with which the centre of the pupil ("pupil-C") moves.
velocity_dict = {}
for da_name, da in position_norm_dict.items():
    velocity_dict[da_name] = kin.compute_velocity(da.sel(keypoints="pupil-C"))

# %%
# Now we can plot pupil velocity over time.
fig, axes = plt.subplots(2, 1, figsize=(6, 4))
for i, (da_name, da) in enumerate(velocity_dict.items()):
    for coord, color in ["x", "C0"], ["y", "C1"]:
        axes[i].plot(
            time_points,
            da.sel(space=coord, time=time_points),
            label=coord,
            color=color,
        )
    axes[i].set_xlabel("Time (s)")
    axes[i].set_ylabel("Velocity")
    axes[i].set_title(f"Pupil velocity ({da_name})")
    axes[i].legend()

plt.tight_layout()
plt.show()
# %%
# The positive peaks correspond to rapid eye movements to the right, the
# negative peaks correspond to rapid eye movements to the left.
# %%
# Pupil diameter
# --------------
# In these datasets, the distance between the two pupil keypoints
# is used to quantify the pupil diameter.
pupil_diameter_dict = {}
for da_name, da in position_norm_dict.items():
    left = da.sel(keypoints="pupil-L").squeeze()
    right = da.sel(keypoints="pupil-R").squeeze()
    dx = right.sel(space="x") - left.sel(space="x")
    dy = right.sel(space="y") - left.sel(space="y")
    # TODO: can this be done with compute_pairwise_distances from kinematics?
    pupil_diameter_dict[da_name] = (dx**2 + dy**2) ** 0.5  # euclidean distance

fig, ax = plt.subplots(figsize=(5, 3))
for da_name, da in pupil_diameter_dict.items():
    ax.plot(da.time, da, label=da_name)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pupil Diameter (pixels)")
ax.legend()
ax.set_title("Pupil Diameter")
plt.tight_layout()
plt.show()

# %%
# Pupil Diameter after filter
# ---------------------------
# A filter can be used to smooth out pupil size data. Unlike eye movements,
# which can be extremely fast, pupil size is unlikely to change rapidly. A
# Moving Average Filter is used here to smooth the data by averaging a
# specified number of data points (defined by the window size) to reduce noise.

# TODO: Use movement filter!
filter = 150  # number of frames over which the filter is used
fig, ax = plt.subplots(figsize=(5, 3))
for da_name, da in pupil_diameter_dict.items():
    filtered_data = np.convolve(da, np.ones(filter) / filter, mode="same")
    # remove the first and last timepoints that are distorted by the filter
    plot_slice = slice(filter // 2, -filter // 2)
    ax.plot(
        da.coords["time"][plot_slice],
        filtered_data[plot_slice],
        label=da_name + " (filter)",
    )
ax.set(
    xlabel="Time (frames)",
    ylabel="Pupil Diameter (pixels)",
    title="Filtered Pupil Diameter",
)
ax.legend()
plt.tight_layout()
plt.show()

# %%
