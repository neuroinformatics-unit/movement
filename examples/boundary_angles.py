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
import numpy as np
import xarray as xr

from movement import sample_data
from movement.plots import plot_trajectory
from movement.roi import PolygonOfInterest

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
# "octadecagonal doughnut".

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
centre = np.array([712.5, 541])
width = 22
extent = 1078  # 1080 - 2 or 1247-178, there's some give-and-take

# Create the vertices of a "unit" octadecagon, centred on (0,0)
n_pts = 18
unit_shape = np.array(
    [
        np.exp((np.pi / 2.0 + (2.0 * i * np.pi) / n_pts) * 1.0j)
        for i in range(n_pts)
    ],
    dtype=complex,
)
# Then stretch and translate the reference to match our arena
outer_boundary = extent / 2.0 * unit_shape.copy()
outer_boundary = (
    np.array([outer_boundary.real, outer_boundary.imag]).transpose() + centre
)
inner_boundary = (extent - width) / 2.0 * unit_shape.copy()
inner_boundary = (
    np.array([inner_boundary.real, inner_boundary.imag]).transpose() + centre
)

# Create our Region of Interest from the outer boundary points, and include a
# hole defined by the inner boundary points.
arena = PolygonOfInterest(outer_boundary, holes=[inner_boundary], name="Arena")

# %%
# We can plot what we have created to provide a visual aid.
fig, ax = plt.subplots(1, 1)

# Plot the RegionOfInterest
ax.plot(
    [c[0] for c in arena.exterior_boundary.coords],
    [c[1] for c in arena.exterior_boundary.coords],
    "black",
    lw=0.5,
)
ax.plot(
    [c[0] for c in arena.interior_boundaries[0].coords],
    [c[1] for c in arena.interior_boundaries[0].coords],
    "black",
    lw=0.5,
)
img = plt.imread(
    "/home/ccaegra/.movement/data/frames/three-mice_Aeon_frame-5sec.png"
)
ax.imshow(img)

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

arena = PolygonOfInterest(outer_boundary, holes=[inner_boundary], name="Arena")

d = arena.contains_point(positions)

# %%
