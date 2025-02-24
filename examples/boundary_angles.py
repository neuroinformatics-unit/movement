"""Compute Angles Relative to Regions of Interest
=================================================

Compute egocentric and allocentric boundary angles,
using ``movement``'s Regions of Interest.
"""

# %%
# Imports
# -------

# For interactive plots: install ipympl with `pip install ipympl` and uncomment
# the following lines in your notebook
# %matplotlib widget
import matplotlib.pyplot as plt
import xarray as xr

from movement import sample_data
from movement.plots import plot_trajectory

# %%
# Load Sample Dataset
# -------------------
# First, we load the ``SLEAP_three-mice_Aeon_proofread`` example dataset.
# For the rest of this example we'll only need the ``position`` data array, so
# we store it in a separate variable.

ds = sample_data.fetch_dataset("SLEAP_three-mice_Aeon_proofread.analysis.h5")
positions: xr.DataArray = ds.position
# %%
# The individuals in this dataset follow very similar, arc-like trajectories.
# This is because they are actually confined to an arena in the shape of an
# "octagonal doughnut".
# We can define the arena as a region of interest to aid with our analysis of
# the animals' behaviours. In this example, we are interested in the egocentric
# and allocentric angles between the individuals' forward direction (their
# direction of travel) and their approach to the closest part of the enclosure
# wall.

fig, ax = plt.subplots(1, 1)
for mouse_name, col in zip(
    positions.individuals.values, ["r", "g", "b"], strict=False
):
    plot_trajectory(
        positions,
        individual=mouse_name,
        keypoints="centroid",
        ax=ax,
        linestyle="-",
        marker=".",
        s=2,
        linewidth=0.5,
        c=col,
        label=mouse_name,
    )
ax.invert_yaxis()
ax.set_title("Trajectories")
ax.set_xlabel("x (pixels)")
ax.set_ylabel("y (pixels)")
ax.legend()

# %%
# Define Regions of Interest
# --------------------------
# First, we should define our arena as a region of interest. Since our arena is
# two-dimensional, we use a ``PolygonOfInterest`` to describe it
# programmatically.
# The future napari plugin will support creating regions of interest by
# clicking points and drawing shapes in the napari GUI. For the time being, we
# can still define our arena by specifying the points that make up the interior
# and exterior boundaries.
