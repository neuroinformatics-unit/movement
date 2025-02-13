"""Analyse eye movements of a mouse
===================================

Look at eye movements caused by the oculo-motor reflex.
"""

# %%
# Imports
# -------
from pathlib import Path

import sleap_io as sio
from matplotlib import pyplot as plt

# from movement import sample_data
from movement.io import load_poses
from movement.plots import plot_trajectory
from movement.utils.vector import compute_norm

# %%
# Load the data
# -------------
# Define the file paths to the data.

data_dir = Path.home() / "Data" / "Sepi-data" / "eye tracking"
video_path = data_dir / "CA514_C_day2_black.avi"
dlc_predictions_path = (
    data_dir
    / "CA514_C_day2_blackDLC_resnet50_eye-trackingFeb19shuffle1_500000.h5"
)


# %%
# Lazy-load the video using sleap_io

video = sio.load_video(video_path)
n_frames, height, width, channels = video.shape

print(f"Number of frames: {n_frames}")
print(f"Frame size: {width}x{height}")
print(f"Number of channels: {channels}")

# %%
# Load the data into movement

ds = load_poses.from_dlc_file(dlc_predictions_path, fps=40)
ds

# %%
# Plot the trajectory of pupil midpoint

fig, ax = plt.subplots()
# Load the 100-th frame of the video as background
ax.imshow(video[100], cmap="Greys_r")
plot_trajectory(ds.position, ax=ax, keypoints=["pupil-L", "pupil-R"])


# %%
position = ds.position.squeeze()

# %%


def quick_plot(da, time=None):
    selection = dict(space="x")
    if time:
        selection["time"] = slice(*time)
    da = da.squeeze()
    da.sel(**selection, drop=True).plot.line(
        x="time", hue="keypoints", aspect=3, size=3
    )


# %%
eye_midpoint = ds.position.sel(keypoints=["eye-L", "eye-R"]).mean("keypoints")
ds["position_subtracted"] = ds.position - eye_midpoint
quick_plot(ds.position)
quick_plot(ds.position_subtracted)

# %%
eye_midpoint = ds.position.sel(keypoints=["eye-L", "eye-R"]).mean("keypoints")
ds["position_subtracted"] = ds.position - eye_midpoint
eye_midpoint.squeeze().plot.line(x="time", hue="space", aspect=2, size=2.5)
quick_plot(ds.position)
quick_plot(ds.position_subtracted)

# %%
eye_length = compute_norm(
    ds.position.sel(keypoints="eye-L") - ds.position.sel(keypoints="eye-R")
)


# %%
pupil_center = ds.position_subtracted.sel(
    keypoints=["pupil-L", "pupil-R"]
).mean("keypoints")
quick_plot(pupil_center, time=(25, 50))

# %%
