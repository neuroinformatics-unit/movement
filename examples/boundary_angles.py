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
#
# The individuals in this dataset follow very similar, arc-like trajectories.
# This is because they are actually confined to an arena in the shape of an
# "octadecagonal doughnut"; an 18-sided regular polygon, with a smaller
# 18-sided regular polygon "cut out" of it.

ds = sample_data.fetch_dataset("SLEAP_three-mice_Aeon_proofread.analysis.h5")
positions: xr.DataArray = ds.position

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
width = 40.0  # The "ring" is 40 pixels wide
extent = 1090.0  # Distance between opposite vertices

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
# View Individual Paths inside the Arena
# --------------------------------------
# We can plot what we have created to provide a visual aid.
fig, ax = plt.subplots(1, 1)

# Plot the arena
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
# Overlay an image of the experimental arena
frame_path = sample_data.fetch_dataset_paths(
    "SLEAP_three-mice_Aeon_proofread.analysis.h5"
)["frame"]
ax.imshow(plt.imread(frame_path))

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
        s=1,
        c=col,
        label=mouse_name,
    )
ax.invert_yaxis()
ax.set_title("Trajectories")
ax.set_xlabel("x (pixels)")
ax.set_ylabel("y (pixels)")
ax.legend()

# %%
# And we can confirm that all individuals, at all timepoints, are indeed
# contained within our arena.
were_inside = arena.contains_point(positions).all()

if were_inside:
    print(
        "All recorded positions, at all times, "
        "and for all individuals, were inside the arena."
    )
else:
    print("At least one position was recorded outside the arena.")

# %%
# Compute Distance to Arena Boundary
# ----------------------------------
# Having defined our region of interest, we can begin to ask questions about
# the individual's behaviour during the experiment.
# One thing of particular interest to us might be the distance between an
# individual and the arena boundary. We can query the ``PolygonOfInterest``
# that we created for this information.

distances_to_boundary = arena.compute_distance_to(
    positions, boundary_only=True
)
distances_fig, distances_ax = plt.subplots(1, 1)
for mouse_name, col in zip(
    positions.individuals.values, ["r", "g", "b"], strict=False
):
    distances_ax.plot(
        distances_to_boundary.sel(individuals=mouse_name),
        c=col,
        label=mouse_name,
    )
distances_ax.legend()
distances_ax.set_xlabel("Time (frames)")
distances_ax.set_ylabel("Distance to closest arena wall (pixels)")
distances_fig.show()

# %%
# ``compute_distance_to`` is returning the distance to the closest
# point on the arena's boundary, which could be either the interior or exterior
# boundary. If we wanted to explicitly know the distance to the exterior
# boundary, for example, we can call this on the corresponding attribute of our
# Polygon.

distances_to_exterior = arena.exterior_boundary.compute_distance_to(positions)
distances_exterior_fig, distances_exterior_ax = plt.subplots(1, 1)
for mouse_name, col in zip(
    positions.individuals.values, ["r", "g", "b"], strict=False
):
    distances_exterior_ax.plot(
        distances_to_exterior.sel(individuals=mouse_name),
        c=col,
        label=mouse_name,
    )
distances_exterior_ax.legend()
distances_exterior_ax.set_xlabel("Time (frames)")
distances_exterior_ax.set_ylabel("Distance to exterior wall (pixels)")
distances_exterior_fig.show()

# %%
# Note that we can also obtain the approach vector for each individual, at each
# time-point as well, using the ``compute_approach_vector`` method.

approach_vectors = arena.compute_approach_vector(positions, boundary_only=True)

# %%
# The ``boundary_only`` Keyword
# -----------------------------
# The ``boundary_only`` argument that is passed to ``compute_distance_to``
# should be noted. By passing this argument as ``True``, we force ``movement``
# to compute the distance to the boundary of our region of interest. If we
# passed this value as ``False``, then ``movement`` would treat points inside
# the region as being a distance 0 from the region - and so the resulting
# distances would all be 0!

# The ``boundary_only`` keyword argument can be provided to a number of methods
# of the ``PolygonOfInterest`` and ``LineOfInterest`` classes. In each case,
# the effect is the same; to toggle whether interior points of the region are
# considered part of the region (``boundary_only = False``) or not
# (``boundary_only = True``). Care should be taken when dealing with 1D regions
# of interest (EG segments, walls), since the "boundary" of a 1D object is
# considered to be just the endpoints.

# %%
# Boundary Angles
# ---------------
# Having observed the individuals' behaviour, we can begin to ask questions
# about their orientation with respect to the boundary. ``movement`` currently
# supports the computation of two such "boundary angle"s;
#
# - The allocentric boundary angle. Given a region of interest $R$, reference
#   vector $\vec{r}$ (such as global north), and a position $p$ in space (EG
#   the position of an individual), the allocentric boundary angle is the
#   signed angle between the approach vector from $p$ to $R$, and $\vec{r}$.
# - The egocentric boundary angle. Given a region of interest $R$, a forward
#   vector $\vec{f}$ (EG the direction of travel of an individual), and the
#   point of origin of $\vec{f}$ denoted $p$ (EG the individual's current
#   position), the egocentric boundary angle is the signed angle between the
#   approach vector from $p$ to $R$ and $\vec{f}$.
#
#
# Note that egocentric angles are generally computed with changing frames of
# reference in mind - the forward vector may be varying in time as the
# individual moves around the arena. By contrast, allocentric angles are always
# computed with respect to some fixed reference frame.
#
# For the purposes of our example, we will define our "forward vector" as the
# displacement vector between successive time-points, for each individual.
# Furthermore, recall that as the individual moves through the arena, the
# "nearest wall" may switch from the inside wall to the outside wall. This
# causes abrupt changes in the approach vector (and hence, computed angle).
# With this in mind, we will only consider the inner wall in the following
# calculations.

forward_vector = positions.diff(dim="time", label="lower")
global_north = np.array([1.0, 0.0])

inner_wall = arena.interior_boundaries[0]
allocentric_angles = inner_wall.compute_allocentric_angle_to_nearest_point(
    positions,
    reference_vector=global_north,
    boundary_only=False,
    in_degrees=True,
)
egocentric_angles = inner_wall.compute_egocentric_angle_to_nearest_point(
    forward_vector,
    positions[:-1],
    boundary_only=False,
    in_degrees=True,
)
# %%
# Can can plot the evolution of these two angular quantities on the same axis.

angle_plot, angle_ax = plt.subplots(2, 1, sharex=True)
allo_ax, ego_ax = angle_ax

for mouse_name, col in zip(
    positions.individuals.values, ["r", "g", "b"], strict=False
):
    allo_ax.plot(
        allocentric_angles.sel(individuals=mouse_name),
        c=col,
        label=mouse_name,
    )
    ego_ax.plot(
        egocentric_angles.sel(individuals=mouse_name),
        c=col,
        label=mouse_name,
    )

ego_ax.set_xlabel("Time (frames)")

ego_ax.set_ylabel("Egocentric angle (degrees)")
ego_ax.set_ylim(-180, 180)
allo_ax.set_ylabel("Allocentric angle (degrees)")
allo_ax.set_ylim(-180, 180)
allo_ax.legend()

angle_plot.tight_layout()
angle_plot.show()

# %%
# We observe that the allocentric angles display a step-like behaviour. This
# is because the interior wall is a series of straight line segments, so the
# approach vector to the wall is constant as the individual travels parallel
# to that segment of wall. As the individual passes a corner in the wall, the
# angle varies rapidly as the approach vector rotates with the individual's
# position, before becoming constant again as the individual resumes motion
# parallel to the next segment of the enclosure wall.
#
# By contrast, the egocentric angles spend long periods of time fluctuating
# near a particular angle, occasionally rapidly changing angle sign. This
# indicates that, by-and-large, the "forward" direction of the individuals is
# remaining constant relative to the interior wall, with the fluctuations
# attributable to small deviations in the direction of travel. The "flipping"
# of the angle sign indicates that the individual undertook a U-turn. That is,
# switched from travelling clockwise around the enclosure to anticlockwise
# (or vice-versa).
