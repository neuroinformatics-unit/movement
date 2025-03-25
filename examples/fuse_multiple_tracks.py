"""Fuse multiple tracking sources
============================

Demonstrate how to combine tracking data from multiple sources to produce a more
accurate trajectory. This is particularly useful in cases where different tracking
methods may fail in different situations, such as with ID swaps.
"""

# %%
# Imports
# -------

from matplotlib import pyplot as plt

from movement import sample_data
from movement.io import load_poses
from movement.plots import plot_centroid_trajectory
from movement.track_fusion import fuse_tracks

# %%
# Load sample datasets
# -------------------
# We'll load the DeepLabCut and SLEAP data for the same mouse in an EPM (Elevated Plus Maze)
# experiment. The DeepLabCut data is considered more reliable, while the SLEAP data was
# generated using a model trained on less data.

# DeepLabCut data (considered more reliable)
dlc_path = sample_data.fetch_dataset_paths("DLC_single-mouse_EPM.predictions.h5")["poses"]
ds_dlc = load_poses.from_dlc_file(dlc_path, fps=30)

# SLEAP data (considered less reliable)
sleap_path = sample_data.fetch_dataset_paths("SLEAP_single-mouse_EPM.analysis.h5")["poses"]
ds_sleap = load_poses.from_sleap_file(sleap_path, fps=30)

# %%
# Inspect the datasets
# -------------------
# Let's look at the available keypoints in each dataset.

print("DeepLabCut keypoints:", ds_dlc.keypoints.values)
print("SLEAP keypoints:", ds_sleap.keypoints.values)

# %%
# The two datasets might have different keypoints, so we'll focus on the centroid.
# If "centroid" doesn't exist in one of the datasets, we would need to compute it
# from other keypoints or choose a different keypoint common to both datasets.

# %%
# Visualize the tracking from the individual sources
# -------------------------------------------------
# First let's plot the centroid trajectory from both sources separately.

# Create a figure with two subplots side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot DLC trajectory
plot_centroid_trajectory(ds_dlc.position, ax=axes[0])
axes[0].set_title('DeepLabCut Tracking')
axes[0].invert_yaxis()  # Invert y-axis to match image coordinates

# Plot SLEAP trajectory
plot_centroid_trajectory(ds_sleap.position, ax=axes[1])
axes[1].set_title('SLEAP Tracking')
axes[1].invert_yaxis()

fig.tight_layout()

# %%
# Fuse tracks using different methods
# -----------------------------------
# Now we'll combine the tracks using different fusion methods and compare the results.

# List of methods to try
methods = ["mean", "median", "weighted", "reliability", "kalman"]

# Create figure with 3 subplots (3 rows, 2 columns)
fig, axes = plt.subplots(3, 2, figsize=(12, 15))
axes = axes.flatten()

# Plot the original tracks in the first two subplots
plot_centroid_trajectory(ds_dlc.position, ax=axes[0])
axes[0].set_title('Original: DeepLabCut')
axes[0].invert_yaxis()

plot_centroid_trajectory(ds_sleap.position, ax=axes[1])
axes[1].set_title('Original: SLEAP')
axes[1].invert_yaxis()

# Fuse and plot the tracks with different methods
for i, method in enumerate(methods, 2):
    if i < len(axes):
        # Set weights for weighted method (example: 0.7 for DLC, 0.3 for SLEAP)
        weights = [0.7, 0.3] if method == "weighted" else None
        
        # Fuse the tracks
        fused_track = fuse_tracks(
            datasets=[ds_dlc, ds_sleap],
            method=method,
            keypoint="centroid",
            weights=weights,
            print_report=True
        )
        
        # Plot the fused track
        plot_centroid_trajectory(fused_track, ax=axes[i])
        axes[i].set_title(f'Fused: {method.capitalize()}')
        axes[i].invert_yaxis()

fig.tight_layout()

# %%
# Detailed Comparison: Kalman Filter Fusion
# ----------------------------------------
# Let's take a closer look at the Kalman filter method, which often provides
# good results for trajectory data.

# Create a new figure
plt.figure(figsize=(10, 8))

# Fuse tracks with Kalman filter
kalman_fused = fuse_tracks(
    datasets=[ds_dlc, ds_sleap],
    method="kalman",
    keypoint="centroid",
    process_noise_scale=0.01,  # Controls smoothness of trajectory
    measurement_noise_scales=[0.1, 0.3],  # Lower values for more reliable sources
    print_report=True
)

# Plot trajectories from both sources and the fused result
plt.plot(
    ds_dlc.position.sel(keypoints="centroid", space="x"),
    ds_dlc.position.sel(keypoints="centroid", space="y"),
    'b.', alpha=0.5, label='DeepLabCut'
)
plt.plot(
    ds_sleap.position.sel(keypoints="centroid", space="x"),
    ds_sleap.position.sel(keypoints="centroid", space="y"),
    'g.', alpha=0.5, label='SLEAP'
)
plt.plot(
    kalman_fused.sel(space="x"),
    kalman_fused.sel(space="y"),
    'r-', linewidth=2, label='Kalman Fused'
)

plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3)
plt.legend()
plt.title('Comparison of Original Tracks and Kalman-Fused Track')
plt.xlabel('X Position')
plt.ylabel('Y Position')

# %%
# Temporal Analysis: Plotting Coordinate Values Over Time
# ------------------------------------------------------
# Let's look at how the x-coordinate values change over time for the different sources.

# Create a new figure
plt.figure(figsize=(12, 6))

# Plot x-coordinate over time
time_values = kalman_fused.time.values

plt.plot(
    time_values,
    ds_dlc.position.sel(keypoints="centroid", space="x"),
    'b-', alpha=0.5, label='DeepLabCut'
)
plt.plot(
    time_values,
    ds_sleap.position.sel(keypoints="centroid", space="x"),
    'g-', alpha=0.5, label='SLEAP'
)
plt.plot(
    time_values,
    kalman_fused.sel(space="x"),
    'r-', linewidth=2, label='Kalman Fused'
)

plt.grid(True, alpha=0.3)
plt.legend()
plt.title('X-Coordinate Values Over Time')
plt.xlabel('Time')
plt.ylabel('X Position')

# %%
# Multiple-Animal Tracking Example with Potential ID Swaps
# -------------------------------------------------------
# Now let's look at a more complex example with multiple animals,
# where ID swaps might be an issue.
# For this, we'll use the SLEAP datasets for three mice.

# Load the two SLEAP datasets with three mice
ds_proofread = sample_data.fetch_dataset(
    "SLEAP_three-mice_Aeon_proofread.analysis.h5"
)
ds_mixed = sample_data.fetch_dataset(
    "SLEAP_three-mice_Aeon_mixed-labels.analysis.h5"
)

print("Proofread dataset individuals:", ds_proofread.individuals.values)
print("Mixed-labels dataset individuals:", ds_mixed.individuals.values)

# %%
# For each individual in the dataset, fuse the tracks from both sources

# Create a figure for comparing original and fused tracks
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Flatten axes for easier iteration
axes = axes.flatten()

# Plot the original tracks for each mouse in the first row
for i, individual in enumerate(ds_proofread.individuals.values):
    if i < 3:  # First row
        # Plot original trajectory from proofread dataset (more reliable)
        pos = ds_proofread.position.sel(individuals=individual)
        plot_centroid_trajectory(pos, ax=axes[i])
        axes[i].set_title(f'Original: {individual}')
        axes[i].invert_yaxis()

# Fuse and plot the tracks for each mouse in the second row
for i, individual in enumerate(ds_proofread.individuals.values):
    if i < 3:  # We have 3 mice
        # Get the individual datasets
        individual_ds_proofread = ds_proofread.sel(individuals=individual)
        individual_ds_mixed = ds_mixed.sel(individuals=individual)
        
        # Fuse the tracks with the Kalman filter (can be replaced with other methods)
        fused_track = fuse_tracks(
            datasets=[individual_ds_proofread, individual_ds_mixed],
            method="kalman",
            keypoint="centroid",
            # More weight to the proofread dataset (considered more reliable)
            measurement_noise_scales=[0.1, 0.3],
            print_report=False
        )
        
        # Plot the fused track
        plot_centroid_trajectory(fused_track, ax=axes[i+3])
        axes[i+3].set_title(f'Fused: {individual}')
        axes[i+3].invert_yaxis()

fig.tight_layout()

# %%
# Conclusions
# ----------
# We've demonstrated several methods for combining tracking data from multiple sources:
#
# 1. **Mean**: Simple averaging of all valid measurements.
# 2. **Median**: More robust to outliers than the mean.
# 3. **Weighted**: Weighted average based on source reliability.
# 4. **Reliability-based**: Selects the most reliable source at each time point.
# 5. **Kalman filter**: Probabilistic approach that models position and velocity.
#
# The Kalman filter often provides the best results as it can:
# - Handle noisy measurements from multiple sources
# - Model the dynamics of movement (position and velocity)
# - Provide smooth trajectories that follow physical constraints
# - Handle missing data and uncertainty in measurements
#
# For multi-animal tracking with potential ID swaps, track fusion can be particularly
# valuable. By combining information from different tracking methods that may fail in
# different situations, we can produce more accurate trajectories across time. 