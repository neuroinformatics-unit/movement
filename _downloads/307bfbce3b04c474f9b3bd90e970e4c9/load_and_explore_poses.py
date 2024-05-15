"""Load and explore pose tracks
===============================

Load and explore an example dataset of pose tracks.
"""

# %%
# Imports
# -------
from matplotlib import pyplot as plt

from movement import sample_data
from movement.io import load_poses

# %%
# Define the file path
# --------------------
# This should be a file output by one of our supported pose estimation
# frameworks (e.g., DeepLabCut, SLEAP), containing predicted pose tracks.
# For example, the path could be something like:

# uncomment and edit the following line to point to your own local file
# file_path = "/path/to/my/data.h5"

# %%
# For the sake of this example, we will use the path to one of
# the sample datasets provided with ``movement``.

file_path = sample_data.fetch_dataset_paths(
    "SLEAP_three-mice_Aeon_proofread.analysis.h5"
)["poses"]
print(file_path)

# %%
# Load the data into movement
# ---------------------------

ds = load_poses.from_sleap_file(file_path, fps=50)
print(ds)

# %%
# The loaded dataset contains two data variables:
# ``position`` and ``confidence``.
# To get the position data:
position = ds.position

# %%
# Select and plot data with xarray
# --------------------------------
# You can use the ``sel`` method to index into ``xarray`` objects.
# For example, we can get a ``DataArray`` containing only data
# for a single keypoint of the first individual:

da = position.sel(individuals="AEON3B_NTP", keypoints="centroid")
print(da)

# %%
# We could plot the x, y coordinates of this keypoint over time,
# using ``xarray``'s built-in plotting methods:
da.plot.line(x="time", row="space", aspect=2, size=2.5)

# %%
# Similarly we could plot the same keypoint's x, y coordinates
# for all individuals:

da = position.sel(keypoints="centroid")
da.plot.line(x="time", row="individuals", aspect=2, size=2.5)

# %%
# Trajectory plots
# ----------------
# We are not limited to ``xarray``'s built-in plots.
# For example, we can use ``matplotlib`` to plot trajectories
# (using scatter plots):

mouse_name = "AEON3B_TP1"

plt.scatter(
    da.sel(individuals=mouse_name, space="x"),
    da.sel(individuals=mouse_name, space="y"),
    s=2,
    c=da.time,
    cmap="viridis",
)
plt.title(f"Trajectory of {mouse_name}")
plt.xlabel("x")
plt.ylabel("y")
plt.colorbar(label="time (sec)")
