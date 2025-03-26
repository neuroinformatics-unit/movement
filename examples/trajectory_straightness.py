"""Trajectory straightness analysis
==============================

This example demonstrates how to compute and visualize the straightness index
of animal movement trajectories.
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
from movement.trajectory_complexity import compute_straightness_index
from movement.plots import plot_centroid_trajectory

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
# We'll use the position data for our trajectory analysis
position = ds.position

# %% 
# Plot trajectories
# ----------------
# First, let's visualize the trajectories of the mice in the XY plane,
# to get a sense of their movement patterns.

fig, ax = plt.subplots(1, 1, figsize=(10, 8))
# Plot trajectories for each mouse
for mouse_name, col in zip(
    position.individuals.values,
    ["r", "g", "b"],  # colors
    strict=False,
):
    plot_centroid_trajectory(
        position,
        individual=mouse_name,
        ax=ax,  # Use the same axes for all plots
        c=col,
        marker="o",
        s=10,
        alpha=0.2,
        label=mouse_name,
    )
    ax.legend().set_alpha(1)

ax.set_title("Mouse Trajectories")
ax.invert_yaxis()  # Make y-axis match image coordinates (0 at top)
plt.tight_layout()

# %% 
# Compute Straightness Index
# -------------------------
# Now let's compute the straightness index for each mouse trajectory. 
# The straightness index is a measure of how direct a path is, calculated as 
# the ratio of the straight-line distance between start and end points to 
# the total path length traveled.
#
# Values close to 1 indicate nearly straight paths, while values closer to 0 
# indicate more tortuous or winding paths.

straightness = compute_straightness_index(position)
print("Straightness index by individual:")
for ind in straightness.individual.values:
    print(f"  {ind}: {straightness.sel(individuals=ind).item():.3f}")

# %%
# Visualize Trajectories with Straightness Index
# ---------------------------------------------
# Let's create a more informative plot that shows both the trajectories
# and their straightness index values.

fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# Plot trajectories for each mouse
for mouse_name, col in zip(
    position.individuals.values,
    ["r", "g", "b"],  # colors
    strict=False,
):
    plot_centroid_trajectory(
        position,
        individual=mouse_name,
        ax=ax,
        c=col,
        marker="o",
        s=10,
        alpha=0.2,
    )
    
    # Add a text annotation with the straightness index
    si_value = straightness.sel(individuals=mouse_name).item()
    # Get the starting point of the trajectory
    start_x = position.sel(individuals=mouse_name, space="x").isel(time=0).item()
    start_y = position.sel(individuals=mouse_name, space="y").isel(time=0).item()
    
    # Add label with straightness index
    ax.text(
        start_x, start_y - 30,  # Position the text near the start
        f"{mouse_name}: SI = {si_value:.3f}",
        fontsize=12,
        color=col,
        bbox=dict(facecolor="white", alpha=0.7),
    )

ax.set_title("Mouse Trajectories with Straightness Index (SI)")
ax.invert_yaxis()
plt.tight_layout()

# %%
# Computing Straightness for Different Time Segments
# ------------------------------------------------
# We can also compute straightness for specific time segments of the trajectory.
# This can be useful for analyzing how path complexity changes during different
# experimental phases or time periods.

# Define start and stop times (in seconds)
time_segments = [
    (0, 1),      # First second
    (1, 2),      # Second second
    (2, None),   # Remainder of trajectory
]

# Create a figure with subplots for each time segment
fig, axes = plt.subplots(len(time_segments), 1, figsize=(10, 12))

for i, (start, stop) in enumerate(time_segments):
    ax = axes[i]
    
    # Compute straightness for this segment
    segment_straightness = compute_straightness_index(
        position, start=start, stop=stop
    )
    
    # Title with time segment information
    stop_str = f"{stop}" if stop is not None else "end"
    ax.set_title(f"Time segment: {start} to {stop_str} seconds")
    
    # Plot trajectories for each mouse in this segment
    for mouse_name, col in zip(
        position.individuals.values,
        ["r", "g", "b"],  # colors
        strict=False,
    ):
        # Filter position data to the time segment
        if stop is not None:
            segment_pos = position.sel(
                time=slice(start, stop), individuals=mouse_name
            )
        else:
            segment_pos = position.sel(
                time=slice(start, None), individuals=mouse_name
            )
        
        # Plot the trajectory segment
        ax.scatter(
            segment_pos.sel(space="x"),
            segment_pos.sel(space="y"),
            c=col,
            s=10,
            alpha=0.5,
            label=f"{mouse_name}: SI = {segment_straightness.sel(individuals=mouse_name).item():.3f}"
        )
        
        # Connect points with lines
        ax.plot(
            segment_pos.sel(space="x"),
            segment_pos.sel(space="y"),
            c=col,
            alpha=0.3,
        )
    
    ax.legend()
    ax.invert_yaxis()
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")

plt.tight_layout()

# %%
# Conclusion
# ---------
# The straightness index provides a simple but effective measure of path 
# complexity. Values close to 1 indicate more direct movement, while values
# closer to 0 indicate more tortuous or complex movement patterns.
#
# In this example, we've seen how to:
#
# 1. Compute the straightness index for entire trajectories
# 2. Visualize trajectories alongside their straightness index values
# 3. Analyze straightness for specific time segments
#
# This metric is useful for quantifying movement patterns across different
# experimental conditions or comparing movement behavior between individuals. 