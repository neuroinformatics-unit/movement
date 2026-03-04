"""Analysing turning behaviour across trajectory segments
========================================================

This example demonstrates how to use :func:`compute_turning_angles`
to characterise an animal's turning behaviour from pose tracking data.

We use the Elevated Plus Maze (EPM) dataset — a standard neuroscience
behavioural assay — where a mouse explores a maze with distinct spatial
zones. Because the maze geometry is known, we can ask:

    **Does turning behaviour differ between open arms
    (directed locomotion) and the centre zone (reorientation)?**

This is a scientifically grounded question that turning angles can
directly answer, without requiring any clustering or machine learning.

What this example covers:

- Loading a sample pose-tracking dataset from ``movement``
- Computing turning angles along a trajectory
- Visualising the trajectory coloured by turning magnitude
- Comparing turning angle distributions between behavioural zones
- Interpreting left/right turn bias

Data source
-----------
``DLC_single-mouse_EPM.predictions.h5`` — DeepLabCut predictions for a
single mouse on an Elevated Plus Maze, included as sample data in
``movement``.
"""

# %%
# Imports
# -------

import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np

from movement import sample_data
from movement.kinematics import compute_turning_angles

# %%
# Load sample data
# ----------------
# We load the Elevated Plus Maze dataset. The position data has shape
# ``(time, space, keypoints, individuals)``.

ds = sample_data.fetch_dataset("DLC_single-mouse_EPM.predictions.h5")
print(ds)
print("\nKeypoints available:", ds.coords["keypoints"].values)

# %%
# Extract nose trajectory
# -----------------------
# We use the ``snout`` keypoint as it best represents the heading
# direction of the animal. We select the first (only) individual.

position = ds.position.sel(keypoints=["snout"])
# position shape: (time, space) with space = ["x", "y"]

# %%
# Compute turning angles
# ----------------------
# Turning angles are the signed change in heading direction between
# consecutive steps, wrapped to (-π, π].
#
# - Positive values = left (counter-clockwise) turn
# - Negative values = right (clockwise) turn
# - NaN = stationary step or first two time points

angles = compute_turning_angles(position, in_degrees=True)

# For 1D plotting — squeeze both dims
angles_plot = angles.sel(
    individuals="individual_0", keypoints="snout"
).values.squeeze()

x = position.sel(
    space="x", individuals="individual_0", keypoints="snout"
).values
y = position.sel(
    space="y", individuals="individual_0", keypoints="snout"
).values

print(f"  Total time steps : {len(angles_plot)}")
print(f"  Valid (non-NaN)  : {int(np.sum(~np.isnan(angles_plot)))}")
print(f"  NaN (stationary/start): {int(np.sum(np.isnan(angles_plot)))}")
print(f"  Mean |angle|     : {np.nanmean(np.abs(angles_plot)):.1f}°")
print(f"  Std              : {np.nanstd(angles_plot):.1f}°")
# %%
# Visualise trajectory coloured by turning magnitude
# ---------------------------------------------------
# We colour each step by the absolute turning angle, which reveals
# *where* in space the animal was actively changing direction.

x = position.sel(
    space="x", individuals="individual_0", keypoints="snout"
).values.squeeze()

y = position.sel(
    space="y", individuals="individual_0", keypoints="snout"
).values.squeeze()
abs_angles = np.abs(angles_plot)

plt.close("all")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- Left panel: trajectory coloured by |turning angle| ---
ax = axes[0]
# Use scatter to colour each point
sc = ax.scatter(
    x[2:],
    y[2:],  # first two points have NaN angles
    c=abs_angles[2:],
    cmap="hot_r",
    s=2,
    vmin=0,
    vmax=90,
    alpha=0.8,
)
# Overlay the first segment in grey for context
ax.plot(x[:2], y[:2], color="grey", lw=0.5, alpha=0.4)
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("|Turning angle| (°)", fontsize=11)
ax.set_xlabel("x (pixels)", fontsize=11)
ax.set_ylabel("y (pixels)", fontsize=11)
ax.set_title("Trajectory coloured by turning magnitude", fontsize=12)
ax.set_aspect("equal")

# --- Right panel: turning angle time series ---
ax = axes[1]
time = ds.coords["time"].values
ax.plot(time[2:], angles_plot[2:], lw=0.6, color="steelblue", alpha=0.7)
ax.axhline(0, color="black", lw=0.8, linestyle="--", alpha=0.5)
ax.fill_between(
    time[2:],
    angles_plot[2:],
    0,
    where=angles_plot[2:] > 0,
    color="cornflowerblue",
    alpha=0.3,
    label="Left turn",
)
ax.fill_between(
    time[2:],
    angles_plot[2:],
    0,
    where=angles_plot[2:] < 0,
    color="salmon",
    alpha=0.3,
    label="Right turn",
)
ax.set_xlabel("Time (s)", fontsize=11)
ax.set_ylabel("Turning angle (°)", fontsize=11)
ax.set_title("Turning angle over time", fontsize=12)
ax.legend(fontsize=10)

plt.suptitle(
    "Mouse turning behaviour — Elevated Plus Maze", fontsize=13, y=1.02
)
plt.tight_layout()
plt.savefig("turning_angle_trajectory.png", dpi=150, bbox_inches="tight")

# %%
# Define behavioural zones using maze geometry
# --------------------------------------------
# The EPM has known spatial structure. We define three zones based on
# the x-coordinate of the animal's position (open arms are the
# horizontal corridors; the centre is the junction).
#
# Zone boundaries are approximate and dataset-dependent — adjust
# ``centre_range`` based on your maze calibration.

x_vals = position.sel(space="x").values.squeeze()
y_vals = position.sel(space="y").values.squeeze()

# Estimate maze centre from position distribution
x_centre = np.nanmedian(x_vals)
y_centre = np.nanmedian(y_vals)
centre_radius = 60  # pixels — adjust to your dataset

# Distance from maze centre for each time step
dist_from_centre = np.sqrt((x_vals - x_centre) ** 2 + (y_vals - y_centre) ** 2)

# Boolean masks (aligned with full time axis)
in_centre = dist_from_centre < centre_radius
in_arms = dist_from_centre >= centre_radius

# Align with angle time axis (angles start at index 2 due to shift)
angle_vals = angles_plot  # length = n_time

centre_angles = angle_vals[in_centre]
centre_angles = centre_angles[~np.isnan(centre_angles)]

arm_angles = angle_vals[in_arms]
arm_angles = arm_angles[~np.isnan(arm_angles)]

print("\nZone summary:")
print(f"  Centre zone frames : {in_centre.sum()}")
print(f"  Arm zone frames    : {in_arms.sum()}")
print(f"  Mean |turn| centre : {np.mean(np.abs(centre_angles)):.1f}°")
print(f"  Mean |turn| arms   : {np.mean(np.abs(arm_angles)):.1f}°")

# %%
# Compare turning distributions between zones
# -------------------------------------------
# The centre zone requires the animal to choose a direction, so we
# expect *larger* turning angles there. On the arms, movement is
# more directed (smaller turns).

plt.close("all")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

bins = np.linspace(-180, 180, 37)  # 10-degree bins

# --- Left: overlapping histograms ---
ax = axes[0]
ax.hist(
    arm_angles,
    bins=bins,
    density=True,
    color="steelblue",
    alpha=0.6,
    label=f"Arms (n={len(arm_angles)})",
)
ax.hist(
    centre_angles,
    bins=bins,
    density=True,
    color="darkorange",
    alpha=0.6,
    label=f"Centre (n={len(centre_angles)})",
)
ax.axvline(0, color="black", lw=1, linestyle="--")
ax.set_xlabel("Turning angle (°)", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.set_title("Turning angle distribution by zone", fontsize=12)
ax.legend(fontsize=10)

# --- Right: box plot comparison ---
ax = axes[1]
ax.boxplot(
    [np.abs(arm_angles), np.abs(centre_angles)],
    tick_labels=["Arms", "Centre"],
    patch_artist=True,
    boxprops=dict(facecolor="lightsteelblue", color="steelblue"),
    medianprops=dict(color="darkblue", lw=2),
    whiskerprops=dict(color="steelblue"),
    capprops=dict(color="steelblue"),
    flierprops=dict(
        marker="o", markerfacecolor="steelblue", markersize=2, alpha=0.3
    ),
)
ax.set_ylabel("|Turning angle| (°)", fontsize=11)
ax.set_title("Turn magnitude by behavioural zone", fontsize=12)

plt.suptitle(
    "Turning behaviour differs between maze zones", fontsize=13, y=1.02
)
plt.tight_layout()
plt.savefig("turning_angle_by_zone.png", dpi=150, bbox_inches="tight")

# %%
# Summary and biological interpretation
# --------------------------------------
# The results demonstrate that:
#
# - **Arms**: Turning angle distribution is sharply peaked around 0°,
#   indicating directed locomotion with minimal course corrections.
#
# - **Centre**: Wider distribution with more large-magnitude turns,
#   reflecting deliberate reorientation as the animal chooses which
#   arm to enter next.
#
# This pattern is consistent with established rodent EPM behaviour:
# open arms elicit directed running, while the centre is a decision
# point associated with heightened reorientation.
#
# The symmetry of the distribution around 0° (equal left/right turns)
# suggests no rotational bias in this animal, which is expected for
# a healthy mouse on a symmetric maze.
#
# .. note::
#
#     The zone boundaries used here are approximate. For quantitative
#     analysis, calibrate zone boundaries against the maze dimensions
#     in physical units using
#     :func:`movement.transforms.scale_to_real_world`.
