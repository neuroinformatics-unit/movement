"""Behavioral Trajectory Clustering and Straightness Index Analysis
===============================================================

This example demonstrates several trajectory-based behavioral analytics:

- Loading a sample trajectory dataset from the movement package
    (Elevated Plus Maze, EPM)
- Performing spatial/behavioral clustering on the trajectory using KMeans
- Computing straightness index for trajectory segments
- Extracting and visualizing segments with high/low straightness index
- Summarizing behavioral interpretations

Data source:
    DLC_single-mouse_EPM.predictions.h5 (sample data from movement package)
"""

# %%
# Imports
# -------
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from sklearn.cluster import KMeans

from movement.kinematics import straightness_index
from movement.sample_data import fetch_dataset

# %%
# Load sample trajectory data
# --------------------------
# Fetch a dataset of mouse position predictions for the Elevated Plus Maze
ds = fetch_dataset("DLC_single-mouse_EPM.predictions.h5")
# This is a typical pose tracking structure:
# (time, space, keypoint, individual)
# Extract nose trajectory for first individual/keypoint
x_idx = np.where(ds["space"].values == "x")[0][0]
y_idx = np.where(ds["space"].values == "y")[0][0]
keypoint_idx = 0  # Use 'snout' as the reference point (first keypoint)
individual_idx = 0
x = ds["position"][:, x_idx, keypoint_idx, individual_idx].values
y = ds["position"][:, y_idx, keypoint_idx, individual_idx].values
coords = np.stack([x, y], axis=1)

# %%
# Behavioral clustering using KMeans
# ----------------------------------
# Cluster trajectory points based on spatial position (x, y)
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
labels = kmeans.fit_predict(coords)

# %%
# Visualize behavioral clusters on full trajectory
# -----------------------------------------------
plt.figure(figsize=(8, 6))
for i in range(n_clusters):
    plt.scatter(
        coords[labels == i, 0],
        coords[labels == i, 1],
        s=1,
        label=f"Cluster {i}",
    )
plt.xlabel("x")
plt.ylabel("y")
plt.title("Behavioral Clusters on Trajectory (EPM)")
plt.legend(markerscale=6)
plt.tight_layout()
plt.savefig("behavioral_clusters_epm.png")
plt.show()

# %%
# Compute straightness index for sliding windows along the trajectory
# -------------------------------------------------------------------
# This measures directness of movement within each window.
window = 300  # Number of timepoints per segment
stride = 100  # Slide step
straightness_vals = []
starts = []
for start in range(0, len(x) - window + 1, stride):
    seg_x = x[start : start + window]
    seg_y = y[start : start + window]
    coords_seg = np.stack([seg_x, seg_y], axis=-1)
    traj = xr.DataArray(
        coords_seg,
        dims=["time", "space"],
        coords={"space": ["x", "y"]},
    )
    si = straightness_index(traj).values
    straightness_vals.append(si)
    starts.append(start)

# %%
# Find segments with highest and lowest straightness index
# --------------------------------------------------------
high_idx = np.argmax(straightness_vals)
low_idx = np.argmin(straightness_vals)

# %%
# Visualize the two segments on the full trajectory
# -------------------------------------------------
plt.figure(figsize=(10, 6))
plt.plot(x, y, color="gray", alpha=0.3, label="Full Trajectory")
for idx, color, label in zip(
    [starts[high_idx], starts[low_idx]],
    ["green", "red"],
    ["High SI", "Low SI"],
    strict=True,
):
    seg_x = x[idx : idx + window]
    seg_y = y[idx : idx + window]
    plt.plot(seg_x, seg_y, color=color, linewidth=3, label=f"{label} Segment")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Trajectory Segments: High vs. Low Straightness")
plt.legend()
plt.tight_layout()
plt.savefig("trajectory_segments_comparison.png")
plt.show()

# %%
# Print summary values for high and low straightness
# --------------------------------------------------
print(
    f"High straightness segment index: {high_idx}, "
    f"SI: {straightness_vals[high_idx]:.2f}"
)
print(
    f"Low straightness segment index: {low_idx}, "
    f"SI: {straightness_vals[low_idx]:.2f}"
)

# %%
# Summary and Interpretation
# -------------------------
# - The high straightness segment represents direct movement
#   (target-seeking), often along an arm.
# - The low straightness segment represents exploratory behavior,
#   with more turns and local search.
# - This script provides a reproducible pipeline for behavioral motif
#   discovery and comparison using movement's sample data.
