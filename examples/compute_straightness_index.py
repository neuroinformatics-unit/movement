"""Detecting behavioural transitions with rolling straightness
=============================================================

Straightness index (S = D/L) measures how directly an animal moves
between two points. A value of 1 means perfectly straight; 0 means
the animal returned to where it started.

A single global straightness value per trajectory tells you little
about *when* behaviour changed. This example uses the ``window_size``
parameter to compute a **rolling straightness index** — a time series
that reveals transitions between directed locomotion and local
exploration in a mouse on an Elevated Plus Maze (EPM).

What this example covers:

- Computing global straightness per individual
- Computing rolling straightness using ``window_size``
- Visualising behavioural state transitions over time
- Comparing straightness distributions between maze zones

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

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from movement import sample_data
from movement.kinematics import compute_straightness_index

# %%
# Load sample data
# ----------------
# We load the Elevated Plus Maze dataset and select the snout keypoint,
# which best represents the heading direction of the animal.

ds = sample_data.fetch_dataset("DLC_single-mouse_EPM.predictions.h5")
print(ds)

position = ds.position.sel(keypoints=["snout"])
print(f"\nDataset: {ds.sizes['time']} frames at {ds.attrs['fps']} fps")
print(f"Duration: {ds.coords['time'].values[-1]:.1f} seconds")

# %%
# Global straightness index
# -------------------------
# First, compute the global straightness over the entire recording.
# This gives a single number summarising the whole trajectory.

global_si = compute_straightness_index(position)
global_val = global_si.sel(
    individuals="individual_0", keypoints="snout"
).values.squeeze()

print(f"\nGlobal straightness index: {global_val:.3f}")
print(
    "Interpretation: the mouse travelled "
    f"{global_val * 100:.1f}% as directly as possible "
    "from start to finish."
)

# %%
# Rolling straightness index
# --------------------------
# A single number loses all temporal structure. Using ``window_size``,
# we compute local straightness over sliding windows of 1 second
# (30 frames at 30 fps). This reveals *when* the animal was moving
# directedly vs. foraging locally.
#
# This is computed with a single vectorized call — no Python loops.

fps = int(ds.attrs["fps"])
window_size = fps  # 1-second windows = 30 frames

rolling_si = compute_straightness_index(position, window_size=window_size)

# Extract 1D arrays for plotting
time = ds.coords["time"].values
si_vals = rolling_si.sel(
    individuals="individual_0", keypoints="snout"
).values.squeeze()

x = position.sel(
    space="x", individuals="individual_0", keypoints="snout"
).values.squeeze()
y = position.sel(
    space="y", individuals="individual_0", keypoints="snout"
).values.squeeze()

print(f"\nRolling SI summary (window={window_size} frames):")
print(f"  Valid values : {int(np.sum(~np.isnan(si_vals)))}")
print(f"  Mean SI      : {np.nanmean(si_vals):.3f}")
print(f"  Std          : {np.nanstd(si_vals):.3f}")
print(f"  % time SI>0.8: {np.nanmean(si_vals > 0.8) * 100:.1f}%  (directed)")
print(f" % time SI<0.2: {np.nanmean(si_vals < 0.2) * 100:.1f}%  (exploratory)")

# %%
# Visualise rolling straightness over time
# -----------------------------------------
# We plot the rolling SI time series and colour the trajectory by SI
# to see where directed vs. exploratory movement occurred in space.

# Define thresholds for behavioural states
DIRECTED_THRESHOLD = 0.7
EXPLORATORY_THRESHOLD = 0.3

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- Left: rolling SI time series ---
ax = axes[0]
ax.plot(time, si_vals, lw=0.6, color="grey", alpha=0.5, zorder=1)

# Shade directed locomotion (high SI)
ax.fill_between(
    time,
    si_vals,
    DIRECTED_THRESHOLD,
    where=(si_vals > DIRECTED_THRESHOLD),
    color="steelblue",
    alpha=0.5,
    label=f"Directed (SI>{DIRECTED_THRESHOLD})",
)
# Shade exploratory movement (low SI)
ax.fill_between(
    time,
    si_vals,
    EXPLORATORY_THRESHOLD,
    where=(si_vals < EXPLORATORY_THRESHOLD),
    color="darkorange",
    alpha=0.5,
    label=f"Exploratory (SI<{EXPLORATORY_THRESHOLD})",
)

ax.axhline(
    DIRECTED_THRESHOLD, color="steelblue", lw=1, linestyle="--", alpha=0.7
)
ax.axhline(
    EXPLORATORY_THRESHOLD, color="darkorange", lw=1, linestyle="--", alpha=0.7
)
ax.set_xlabel("Time (s)", fontsize=11)
ax.set_ylabel("Straightness Index", fontsize=11)
ax.set_ylim(-0.05, 1.05)
ax.set_title(
    f"Rolling straightness index\n(window = {window_size} frames = 1 s)",
    fontsize=11,
)
ax.legend(fontsize=9)

# --- Right: trajectory coloured by rolling SI ---
ax = axes[1]
sc = ax.scatter(
    x[window_size:],
    y[window_size:],
    c=si_vals[window_size:],
    cmap="RdYlGn",
    s=2,
    vmin=0,
    vmax=1,
    alpha=0.8,
)
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("Straightness Index", fontsize=10)
ax.set_xlabel("x (pixels)", fontsize=11)
ax.set_ylabel("y (pixels)", fontsize=11)
ax.set_title("Trajectory coloured by straightness", fontsize=11)
ax.set_aspect("equal")

plt.suptitle("Mouse locomotion — Elevated Plus Maze", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("straightness_rolling.png", dpi=150, bbox_inches="tight")
plt.close("all")

# %%
# Compare straightness between maze zones
# ----------------------------------------
# Using maze geometry (distance from centre), we compare SI between
# the open arms and the centre junction.

x_centre = np.nanmedian(x)
y_centre = np.nanmedian(y)
centre_radius = 60  # pixels — adjust to your dataset

dist_from_centre = np.sqrt((x - x_centre) ** 2 + (y - y_centre) ** 2)

in_centre = dist_from_centre < centre_radius
in_arms = dist_from_centre >= centre_radius

# Align masks with si_vals
centre_si = si_vals[in_centre]
centre_si = centre_si[~np.isnan(centre_si)]

arm_si = si_vals[in_arms]
arm_si = arm_si[~np.isnan(arm_si)]

print("\nZone comparison:")
print(f"  Centre zone frames : {in_centre.sum()}")
print(f"  Arm zone frames    : {in_arms.sum()}")
print(f"  Mean SI centre     : {np.mean(centre_si):.3f}")
print(f"  Mean SI arms       : {np.mean(arm_si):.3f}")

# %%
# Visualise zone comparison
# --------------------------

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

bins = np.linspace(0, 1, 26)  # 0.04-wide bins

ax = axes[0]
ax.hist(
    arm_si,
    bins=bins,
    density=True,
    color="steelblue",
    alpha=0.6,
    label=f"Arms (n={len(arm_si)})",
)
ax.hist(
    centre_si,
    bins=bins,
    density=True,
    color="darkorange",
    alpha=0.6,
    label=f"Centre (n={len(centre_si)})",
)
ax.set_xlabel("Straightness Index", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.set_title("Straightness distribution by zone", fontsize=11)
ax.legend(fontsize=10)

ax = axes[1]
ax.boxplot(
    [arm_si, centre_si],
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
ax.set_ylabel("Straightness Index", fontsize=11)
ax.set_title("Straightness by behavioural zone", fontsize=11)

plt.suptitle("Straightness differs between maze zones", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("straightness_by_zone.png", dpi=150, bbox_inches="tight")
plt.close("all")

# %%
# Summary
# -------
# The rolling straightness index reveals temporal structure invisible
# to the global metric:
#
# - **High SI (> 0.7):** Directed locomotion along maze arms
# - **Low SI (< 0.3):** Local exploration, turning, or stationary periods
#
# The zone comparison shows whether directed movement is concentrated
# in the arms (expected for an EPM-experienced mouse) or distributed
# across the maze.
#
# The ``window_size`` parameter is fully vectorized using xarray's
# ``.rolling()`` — computing rolling SI for 100 individuals takes the
# same number of operations as for 1.
#
# .. note::
#
#     The 1-second window (30 frames) is a starting point. Shorter
#     windows (e.g., 10 frames) capture rapid direction changes;
#     longer windows (e.g., 150 frames = 5 s) reflect sustained
#     behavioural states. Choose based on your biological question.
