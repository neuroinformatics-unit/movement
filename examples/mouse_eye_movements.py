"""Analyse eye movements of a mouse
===================================

Look at eye movements caused by the oculo-motor reflex.
"""

# %%
# Imports
# -------
import numpy as np
import sleap_io as sio
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
ds = {"black": ds_black, "uniform": ds_uniform}

# %%
# Explore the data
# -------------
video = {}  # To save videos in
for name, ds_i in ds.items():
    video[name] = sio.load_video(ds_i.video_path)
    n_frames, height, width, channels = video[name].shape
    print(f"Dataset: {name}")
    print(f"Number of frames: {n_frames}")
    print(f"Frame size: {width}x{height}")
    print(f"Number of channels: {channels}\n")

# Both datasets contain videos of a mouse eye under two different conditions.
# %%
# Plot first frame with keypoints
# -------------
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for i, (name, ds_i) in enumerate(ds.items()):
    ax[i].imshow(video[name][0], cmap="gray")  # plot first video frame
    for keypoint in ds_i.position.keypoints.values:
        x = ds_i.position.sel(time=0, space="x", keypoints=keypoint)
        y = ds_i.position.sel(time=0, space="y", keypoints=keypoint)
        ax[i].scatter(x, y, label=keypoint)  # plot keypoints
    ax[i].legend()
    ax[i].set_title(f"{name} (First Frame)")
    ax[i].invert_yaxis()  # because the dataset was collected flipped
plt.tight_layout()
plt.show()

# %%
# Pupil trajectory
# -------------
# A quick trajectory plot of the trajectory of the centre of the pupil.
time_points = ds_black.time[slice(50, 1000)]
da = ds_black.position.sel(time=time_points)  # data array to plot
fig, ax = plot_trajectory(da, keypoints=["pupil-L", "pupil-R"])
fig.show()

# %%
# Pupil trajectories on top of video frame
# -------------
# Plot pupil trajectories for both 'black' and 'uniform' datasets
fig, ax = plt.subplots(1, 2, figsize=(11, 3))
for i, (ds_name, ds_i) in enumerate(ds.items()):
    ax[i].imshow(
        video[ds_name][100], cmap="Greys_r"
    )  # Frame 100 as background
    time_points = ds_i.time[slice(50, 1000)]
    plot_trajectory(
        ds_i.position.sel(time=time_points),
        ax=ax[i],
        keypoints=["pupil-L", "pupil-R"],
        alpha=0.5,
        s=3,
    )
    ax[i].invert_yaxis()
    ax[i].set_title(f"Pupil Trajectory ({ds_name})")
plt.show()


# %%
# Keypoint positions in x and y over time
# -------------
# Helper function to plot the data
def quick_plot(da, time=None, ax=None, **selection):
    if time:
        selection["time"] = slice(*time)
    da = da.squeeze()
    da.sel(**selection, drop=True).plot.line(x="time", ax=ax)


frame_slice = slice(350, 1000)  # frame slice to plot
fig, axes = plt.subplots(2, 2, figsize=(12, 6))
for i, ds_name in enumerate(["uniform", "black"]):
    for j, space in enumerate(["y", "x"]):
        time_points = ds[ds_name].time[frame_slice]
        da = ds[ds_name].position.sel(time=time_points)
        quick_plot(
            da,
            space=space,
            ax=axes[i, j],
        )
        axes[i, j].set_title(f"{ds_name} - {space} position")
for ax in axes.flat:
    ax.set_ylim(100, 500)
plt.tight_layout()
plt.show()

# %%
# Normalise movement to the eye midpoint
# -------------
# Normalizing the pupil's position relative to the eye keypoints reduces the
# impact of head movements or artefacts caused by movement of the camera.

position_norm = {}  # to save the normalised data
for ds_name, ds_i in ds.items():
    eye_midpoint = ds_i.position.sel(keypoints=["eye-L", "eye-R"]).mean(
        "keypoints"
    )
    position_norm[ds_name] = ds_i.position - eye_midpoint

# Plot normalized keypoint x and y coordinates over time
fig, axes = plt.subplots(2, 2, figsize=(12, 6))
for i, ds_name in enumerate(["uniform", "black"]):
    for j, space in enumerate(["y", "x"]):
        time_points = position_norm[ds_name].time[frame_slice]
        quick_plot(
            position_norm[ds_name].sel(time=time_points),
            space=space,
            ax=axes[i, j],
        )
        axes[i, j].set_title(f"{ds_name} (normalized) - {space} position")

# Set the same limits for y-axes
for ax in axes.flat:
    ax.set_ylim(-200, 200)
plt.tight_layout()
plt.show()

# There is less high frequency noise in the signal now.
# %%
# Pupil Centroid
# -------------
# Add a pupil centroid keypoint to position norm

pupil_centroid = {}
for key in ["uniform", "black"]:
    pupil_centroid[key] = (
        position_norm[key]
        .sel(keypoints=["pupil-L", "pupil-R"])
        .mean("keypoints")
    )

# %%
# Pupil position over time
# -------------
# position_norm from the previous example will be used to look at the centroid
# of the pupil and the velocity over time.

fig, axes = plt.subplots(2, 1, figsize=(6, 4))
for i, ds_name in enumerate(["uniform", "black"]):
    time_points = pupil_centroid[ds_name].time[frame_slice].values
    pupil_centroid_window = pupil_centroid[ds_name].sel(time=time_points)
    axes[i].plot(
        time_points,
        pupil_centroid_window.sel(space="x"),
        label="x",
        color="C0",
    )
    axes[i].plot(
        time_points,
        pupil_centroid_window.sel(space="y"),
        label="y",
        color="C1",
    )
    axes[i].set_xlabel("Time (s)")
    axes[i].set_ylabel("Position")
    axes[i].set_title(f"Pupil centroid position ({ds_name})")
    axes[i].legend()

for ax in axes.flat:
    ax.set_ylim(-30, 65)

plt.tight_layout()
plt.show()
# %%
# Pupil velocity over time
# -------------

velocity = {
    ds_name: kin.compute_velocity(pupil_centroid[ds_name])
    for ds_name in pupil_centroid
}

fig, axes = plt.subplots(2, 1, figsize=(6, 4))
for i, ds_name in enumerate(["uniform", "black"]):
    time_points = velocity[ds_name].time[frame_slice].values
    axes[i].plot(
        time_points,
        velocity[ds_name].sel(space="x", time=time_points),
        label="x",
        color="C0",
    )
    axes[i].plot(
        time_points,
        velocity[ds_name].sel(space="y", time=time_points),
        label="y",
        color="C1",
    )
    axes[i].set_xlabel("Time (s)")
    axes[i].set_ylabel("Velocity")
    axes[i].set_title(f"Pupil velocity ({ds_name})")
    axes[i].legend()

plt.tight_layout()
plt.show()

# %%
# Pupil diameter
# -------------
# In these datasets, the distance between the two pupil keypoints
# is used to quantify the pupil diameter.

pupil_diameter = {}
for ds_name, ds_i in ds.items():
    left_pupil = ds_i.position.sel(keypoints="pupil-L").squeeze()
    right_pupil = ds_i.position.sel(keypoints="pupil-R").squeeze()
    dx = right_pupil.sel(space="x") - left_pupil.sel(space="x")  # x difference
    dy = right_pupil.sel(space="y") - left_pupil.sel(space="y")  # y difference
    pupil_diameter[ds_name] = (dx**2 + dy**2) ** 0.5  # euclidean distance

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
for name, da_pupil_diameter in pupil_diameter.items():
    ax.plot(da_pupil_diameter.time, da_pupil_diameter, label=name)
    ax.set_xlabel("Time (frames)")
    ax.set_ylabel("Pupil Diameter (pixels)")
ax.legend()
ax.set_title("Pupil Diameter")
plt.tight_layout()
plt.show()
# %%
# Moving Average Filter
# -------------
# A filter can be used to smooth out pupil size data. Unlike eye movements,
# which can be extremely fast, pupil size is unlikely to change rapidly. A
# Moving Average Filter is used here to smooth the data by averaging a
# specified number of data points (defined by the window size) to reduce noise.

# TODO: Use movement filter!
window_size = 150  # number of frames over which the filter is used
fig, ax = plt.subplots(figsize=(5, 3))
for name, da in pupil_diameter.items():
    filtered_data = np.convolve(
        da, np.ones(window_size) / window_size, mode="same"
    )

    # remove the first and last timepoints that are distorted by the filter
    plot_slice = slice(window_size // 2, -window_size // 2)
    ax.plot(
        da.coords["time"][plot_slice],
        filtered_data[plot_slice],
        label=name + " (filter)",
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
