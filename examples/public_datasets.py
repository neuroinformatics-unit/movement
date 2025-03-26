"""Working with public datasets
==========================

This example demonstrates how to access and work with publicly available
datasets of animal poses and trajectories.
"""

# %%
# Imports
# -------

from movement import public_data
import matplotlib.pyplot as plt

# %%
# Listing available datasets
# -------------------------
# First, let's see what public datasets are available:

datasets = public_data.list_public_datasets()
print("Available public datasets:")
for dataset in datasets:
    info = public_data.get_dataset_info(dataset)
    print(f"\n{dataset}:")
    print(f"  Description: {info['description']}")
    print(f"  URL: {info['url']}")
    print(f"  Paper: {info['paper']}")
    print(f"  License: {info['license']}")

# %%
# CalMS21 Dataset
# --------------
# The CalMS21 dataset contains multi-animal pose tracking data for various
# animal types and behavioral tasks.

# %%
# Let's fetch a subset of the CalMS21 dataset with mice in an open field:

mouse_data = public_data.fetch_calms21(
    subset="train",
    animal_type="mouse",
    task="open_field",
)

# NOTE: This is currently a placeholder implementation.
# In the full implementation, this would download and load actual data.

print("\nDataset attributes:")
for key, value in mouse_data.attrs.items():
    print(f"  {key}: {value}")

# %%
# We can also fetch data for different animal types and tasks:

fly_data = public_data.fetch_calms21(
    subset="train",
    animal_type="fly",
    task="courtship",
)

# NOTE: This is currently a placeholder implementation.
# In the full implementation, this would download and load actual data.

print("\nDataset attributes:")
for key, value in fly_data.attrs.items():
    print(f"  {key}: {value}")

# %%
# Rat7M Dataset
# ------------
# The Rat7M dataset contains tracking data for multiple rats in complex
# environments.

# %%
# Let's fetch a subset of the Rat7M dataset:

rat_data = public_data.fetch_rat7m(subset="open_field")

# NOTE: This is currently a placeholder implementation.
# In the full implementation, this would download and load actual data.

print("\nDataset attributes:")
for key, value in rat_data.attrs.items():
    print(f"  {key}: {value}")

# %%
# Using the data
# -------------
# Once the data is loaded, you can use all the movement functionality
# for analysis and visualization.
#
# NOTE: Since we're currently using placeholder data, we can't demonstrate
# actual analysis here. When the full implementation is complete, this
# example will include code for:
#
# - Visualizing trajectories
# - Computing kinematic measures
# - Analyzing behavioral patterns
# - Comparing across datasets 