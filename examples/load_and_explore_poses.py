"""Load and explore pose data using movement."""


# Load and explore pose tracks
# ===============================
# Load and explore an example dataset of pose tracks.

# %%
# Imports
# -------

import matplotlib.pyplot as plt  # Importing explicitly to keep the plot open

from movement import sample_data
from movement.io import load_poses
from movement.plots import plot_centroid_trajectory

# %%
# Define the file path
# --------------------
# This should be a file output by one of our supported pose estimation
# frameworks (e.g., DeepLabCut, SLEAP), containing predicted pose tracks.

file_path = sample_data.fetch_dataset_paths(
    "SLEAP_three-mice_Aeon_proofread.analysis.h5"
)["poses"]
print("File Path:", file_path)

# %%
# Load the data into movement
# ---------------------------

ds = load_poses.from_sleap_file(file_path, fps=50)
print("Dataset Overview:\n", ds)

# %%
# Extract position data
position = ds.position

# %%
# Select and plot data with xarray
# --------------------------------
# Selecting a single keypoint's data for an individual
da = position.sel(individuals="AEON3B_NTP", keypoints="centroid")
print("Selected Data:\n", da)

# Plot x, y coordinates over time
da.plot.line(x="time", row="space", aspect=2, size=2.5)

# %%
# Plot x, y coordinates for all individuals
da = position.sel(keypoints="centroid")
da.plot.line(x="time", row="individuals", aspect=2, size=2.5)

# %%
# Trajectory plots
# ----------------
kalman-filter-example
# Using movement's built-in trajectory plot function

mouse_name = "AEON3B_TP1"
fig, ax = plot_trajectory(position, individual=mouse_name)

plt.show()  # Ensure the graph stays open
=======
# We are not limited to ``xarray``'s built-in plots.
# The ``movement.plots`` module provides some additional
# visualisations, like ``plot_centroid_trajectory()``.


mouse_name = "AEON3B_TP1"
fig, ax = plot_centroid_trajectory(position, individual=mouse_name)
fig.show()
main
