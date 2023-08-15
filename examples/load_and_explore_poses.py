"""
Load and explore pose tracks
============================

Load and explore an example dataset of pose tracks.
"""

# %%
# Imports
# -------
from matplotlib import pyplot as plt

from movement import datasets
from movement.io import load_poses

# %%
# Fetch an example dataset
# ------------------------
# Print a list of available datasets:

print(datasets.find_pose_data())

# %%
# Fetch the path to an example dataset
# (Feel free to replace this with the path to your own dataset.
# e.g., `file_path = "/path/to/my/data.h5"`)
file_path = datasets.fetch_pose_data_path(
    "SLEAP_three-mice_Aeon_proofread.analysis.h5"
)

# %%
# Load the dataset
# ----------------

ds = load_poses.from_sleap_file(file_path, fps=60)
ds

# %%
# The loaded dataset contains two data variables:
# `pose_tracks` and `confidence`
# To get the pose tracks:
pose_tracks = ds["pose_tracks"]

# %%
# Slect and plot data with ``xarray``
# -----------------------------------
# You can use the ``sel`` method to index into ``xarray`` objects.
# For example, we can get a `DataArray` containing only data
# for the "centroid" keypoint of the first individual:

da = pose_tracks.sel(individuals="AEON3B_NTP", keypoints="centroid")

# %%
# We could plot the x,y coordinates of this keypoint over time,
# using ``xarray``'s built-in plotting methods:
da.plot.line(x="time", row="space", aspect=2, size=2.5)

# %%
# Similarly we could plot the same keypoint's x,y coordinates
# for all individuals:

pose_tracks.sel(keypoints="centroid").plot.line(
    x="time", row="individuals", aspect=2, size=2.5
)

# %%s
# Trajectory plots
# ----------------
# We are not limited to ``xarray``'s built-in plots.
# For example, we can use ``matplotlib`` to plot trajectories
# (using scatter plots):

individuals = pose_tracks.individuals.values
for i, ind in enumerate(individuals):
    da_ind = pose_tracks.sel(individuals=ind, keypoints="centroid")
    plt.scatter(
        da_ind.sel(space="x"),
        da_ind.sel(space="y"),
        s=2,
        color=plt.cm.tab10(i),
        label=ind,
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
