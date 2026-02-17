"""Annotating time with behavioural events
==========================================

Annotate timepoints with behavioural labels and 
use them to select subsets of data.

This example uses a 1-hour recording of a mouse in its home cage
and segments it into active/inactive periods based on a speed threshold.
The key idea is to attach event labels as a **non-dimension coordinate**
along the ``time`` dimension, which lets you filter data by event
using native xarray operations.
"""

# %%
# Imports
# -------

import matplotlib.pyplot as plt
import numpy as np

from movement import sample_data
from movement.kinematics import compute_speed

# %%
# Load the dataset
# ----------------
# This is a DeepLabCut prediction for a single mouse tracked over one hour
# in its home cage.

ds = sample_data.fetch_dataset(
    "DLC_smart-kage3_datetime-20240417T090006.predictions.h5"
)
print(ds)

# %%
# Compute speed
# -------------
# Compute the instantaneous speed for each keypoint.
# The result has dimensions ``(time, keypoints, individuals)``
# (the ``space`` dimension is collapsed into a scalar norm).

speed = compute_speed(ds.position)
print(speed)

# %%
# Derive a single speed per timepoint
# ------------------------------------
# Average speed across all keypoints and individuals to get
# one representative speed value per frame.

mean_speed = speed.mean(dim=["keypoints", "individuals"])

# %%
# Define active vs inactive periods
# ----------------------------------
# Apply a speed threshold to classify each timepoint.
# Frames above the threshold are labelled "active",
# the rest are "inactive".

speed_threshold = 5.0  # units depend on the dataset (pixels/s or mm/s)
event_labels = np.where(mean_speed > speed_threshold, "active", "inactive")

# Attach as a non-dimension coordinate along ``time``
ds = ds.assign_coords(event=("time", event_labels))
print(ds.coords["event"])

# %%
# Select data by event
# --------------------
# With the ``event`` coordinate in place, we can filter
# the dataset using standard xarray boolean indexing.

# Select only "active" timepoints
ds_active = ds.sel(time=ds.event == "active")
print(f"Active frames: {ds_active.sizes['time']}")

# Select only "inactive" timepoints
ds_inactive = ds.sel(time=ds.event == "inactive")
print(f"Inactive frames: {ds_inactive.sizes['time']}")

# %%
# Visualise the segmentation
# --------------------------
# Plot the mean speed over time, coloured by event label.

fig, ax = plt.subplots(figsize=(12, 3))

time = ds.time.values
colors = np.where(event_labels == "active", "tab:orange", "tab:blue")

ax.scatter(time, mean_speed.values, c=colors, s=0.5, alpha=0.7)
ax.axhline(speed_threshold, color="k", linestyle="--", linewidth=0.8)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Mean speed")
ax.set_title("Active (orange) vs Inactive (blue) periods")
fig.tight_layout()
plt.show()

# %%
# Compare trajectories by event
# -----------------------------
# Plot the 2D trajectory of a single keypoint, coloured by event.
# We pick the first keypoint and individual.

keypoint = ds.keypoints.values[0]
individual = ds.individuals.values[0]

pos = ds.position.sel(keypoints=keypoint, individuals=individual)

fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)

for ax, label, color in zip(
    axes, ["active", "inactive"], ["tab:orange", "tab:blue"]
):
    subset = pos.sel(time=ds.event == label)
    ax.scatter(
        subset.sel(space="x"),
        subset.sel(space="y"),
        s=0.3,
        alpha=0.5,
        color=color,
    )
    ax.set_title(f"{label.capitalize()} ({subset.sizes['time']} frames)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")

fig.suptitle(f"Trajectory of '{keypoint}' by event", fontsize=13)
fig.tight_layout()
plt.show()

# %%
# Multiple annotation layers
# --------------------------
# You can attach more than one annotation coordinate to the same
# dimension. For example, you could add a second layer for
# time-of-day or experimental phase, and combine filters:
#
# .. code-block:: python
#
#     ds = ds.assign_coords(phase=("time", phase_labels))
#     ds.sel(time=(ds.event == "active") & (ds.phase == "early"))
#
# This pattern scales naturally to any number of categorical
# annotations without changing the dataset's dimensionality.
