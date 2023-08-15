"""
Load pose tracks
================

Load and explore example dataset of pose tracks.
"""

# %%
# Imports
# -------
from movement import datasets
from movement.io import load_poses

# %%
# Fetch an example dataset
# ------------------------
# Feel free to replace this with the path to your own dataset.
# e.g., `h5_path = "/path/to/my/data.h5"`

h5_path = datasets.fetch_pose_data_path(
    "SLEAP_two-mice_social-interaction.analysis.h5"
)

# %%
# Load the dataset
# ----------------

ds = load_poses.from_sleap_file(h5_path, fps=40)
ds
