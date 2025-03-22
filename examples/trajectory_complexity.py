"""Trajectory complexity measures for animal movement paths.
==========================================================

Compute and visualize various trajectory complexity measures
including straightness index, sinuosity, tortuosity, and more.
"""

# %%
# Imports
# -------

# For interactive plots: install ipympl with `pip install ipympl` and uncomment
# the following line in your notebook
# %matplotlib widget
import numpy as np
from matplotlib import pyplot as plt

from movement import sample_data
from movement.plots import plot_centroid_trajectory
from movement.trajectory_complexity import (
    compute_angular_velocity,
    compute_directional_change,
    compute_sinuosity,
    compute_straightness_index,
    compute_tortuosity,
)

# %%
# Load sample dataset
# ------------------
# First, we load an example dataset. In this case, we select the
# ``SLEAP_three-mice_Aeon_proofread`` sample data.
ds = sample_data.fetch_dataset(
    "SLEAP_three-mice_Aeon_proofread.analysis.h5",
)

print(ds)

# %%
# We'll use the position data for our trajectory complexity analysis
position = ds.position

# %%
# Plot trajectories
# ----------------
# First, let's visualize the trajectories of the mice in the XY plane,
# to get a sense of their movement patterns.

fig, ax = plt.subplots(1, 1, figsize=(10, 8))
plot_centroid_trajectory(
    ds,
    ax=ax,
    color_by="individual",
    plot_markers=False,
    alpha=0.7,
)
ax.set_title("Mouse Trajectories")
ax.invert_yaxis()  # Make y-axis match image coordinates (0 at top)
plt.tight_layout()

# %%
# Straightness Index
# ----------------
# The straightness index is a simple measure of path complexity, defined as the
# ratio of the straight-line distance between start and end points to the total
# path length. Values closer to 1 indicate straighter paths.

straightness = compute_straightness_index(position)
print("Straightness index by individual:")
for ind in straightness.individual.values:
    print(f"  {ind}: {straightness.sel(individual=ind).item():.3f}")

# %%
# Sinuosity
# --------
# Sinuosity provides a local measure of path complexity using a sliding window.
# It's essentially the inverse of straightness - higher values indicate more
# tortuous paths.

sinuosity = compute_sinuosity(position, window_size=20)

# Plot sinuosity over time for each individual
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
for ind in sinuosity.individual.values:
    ax.plot(sinuosity.time, sinuosity.sel(individual=ind), label=ind)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Sinuosity")
ax.set_title("Sinuosity over time (window size = 20 frames)")
ax.legend()
plt.tight_layout()

# %%
# Angular Velocity
# --------------
# Angular velocity measures the rate of change in direction. Higher values
# indicate sharper turns or changes in direction.

ang_vel = compute_angular_velocity(position, in_degrees=True)

# Plot angular velocity over time for each individual
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
for ind in ang_vel.individual.values:
    ax.plot(
        ang_vel.time,
        ang_vel.sel(individual=ind),
        label=ind,
        alpha=0.7,
    )
ax.set_xlabel("Time (s)")
ax.set_ylabel("Angular velocity (degrees)")
ax.set_title("Angular velocity over time")
ax.legend()
plt.tight_layout()

# %%
# Tortuosity
# ---------
# Tortuosity measures the degree of winding or twisting of a path.
# Here we use two different methods: fractal dimension and angular variance.

# Compute tortuosity using angular variance method
tort_ang = compute_tortuosity(position, method="angular_variance")
print("Tortuosity (angular variance) by individual:")
for ind in tort_ang.individual.values:
    print(f"  {ind}: {tort_ang.sel(individual=ind).item():.3f}")

# Compute tortuosity using fractal dimension method
tort_frac = compute_tortuosity(position, method="fractal")
print("\nTortuosity (fractal dimension) by individual:")
for ind in tort_frac.individual.values:
    print(f"  {ind}: {tort_frac.sel(individual=ind).item():.3f}")

# %%
# Directional Change
# ----------------
# Directional change measures the total amount of turning within a window.
# Higher values indicate more meandering behavior.

dir_change = compute_directional_change(
    position, window_size=20, in_degrees=True
)

# Plot directional change over time for each individual
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
for ind in dir_change.individual.values:
    ax.plot(
        dir_change.time,
        dir_change.sel(individual=ind),
        label=ind,
        alpha=0.7,
    )
ax.set_xlabel("Time (s)")
ax.set_ylabel("Directional change (degrees)")
ax.set_title("Directional change over time (window size = 20 frames)")
ax.legend()
plt.tight_layout()

# %%
# Compare measures across individuals
# ---------------------------------
# Let's create a summary bar plot to compare different trajectory complexity measures
# across individuals.

# Collect measures for each individual
individuals = position.individual.values
measures = {
    "Straightness Index": straightness,
    "Tortuosity (Angular)": tort_ang,
    "Tortuosity (Fractal)": tort_frac,
}

# Create bar plot
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
x = np.arange(len(individuals))
width = 0.25
multiplier = 0

for measure_name, measure_data in measures.items():
    offset = width * multiplier
    rects = ax.bar(
        x + offset,
        [measure_data.sel(individual=ind).item() for ind in individuals],
        width,
        label=measure_name,
    )
    multiplier += 1

ax.set_xticks(x + width, individuals)
ax.set_ylabel("Value")
ax.set_title("Comparison of Trajectory Complexity Measures")
ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
plt.tight_layout()

# %%
# Conclusion
# ---------
# These trajectory complexity measures provide different ways to quantify and
# compare animal movement patterns. The choice of measure depends on the specific
# research question:
#
# - **Straightness Index**: Best for overall path directness
# - **Sinuosity**: Good for local path complexity that varies over time
# - **Angular Velocity**: Useful for identifying sharp turns or directional changes
# - **Tortuosity**: Captures the overall winding nature of the path
# - **Directional Change**: Quantifies turning behavior within a time window
#
# By combining these measures, researchers can gain insights into various aspects
# of animal movement behavior.
