"""Compute angles relative to regions of interest
=================================================

Use ``movement``'s Regions of Interest to compute distances from individuals
to points of interest, approach vectors, and both egocentric and allocentric
boundary angles.
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
from movement.plots import plot_centroid_trajectory
from movement.roi import PolygonOfInterest

# %%
# Load Sample Dataset
# -------------------
# In this example, we will use the ``SLEAP_three-mice_Aeon_proofread`` example
# dataset. We only need the ``position`` data array, so we store it in a
# separate variable.

ds = sample_data.fetch_dataset("SLEAP_three-mice_Aeon_proofread.analysis.h5")
positions: xr.DataArray = ds.position

# %%
# The data we have loaded used an arena setup that we have plotted below.

arena_fig, arena_ax = plt.subplots(1, 1)
# Overlay an image of the experimental arena
arena_image = sample_data.fetch_dataset_paths(
    "SLEAP_three-mice_Aeon_proofread.analysis.h5"
)["frame"]
arena_ax.imshow(plt.imread(arena_image))

arena_ax.set_xlabel("x (pixels)")
arena_ax.set_ylabel("y (pixels)")

arena_fig.show()

# %%
# The arena is divided up into three main sub-regions. The cuboidal structure
# on the right-hand-side of the arena is the nest of the three individuals
# taking part in the experiment. The majority of the arena is an open
# octadecagonal (18-sided) shape, which is the bright central region that
# encompasses most of the image. This central region is surrounded by a
# (comparatively thin) "ring", whose interior wall has gaps at regular
# intervals to allow individuals to move from the ring to the centre of the
# region. The nest is also accessible from the ring.
# In this example, we will look at how we can use ``movement``s regions of
# interest (RoIs) to analyse our sample dataset.

# %%
# Define Regions of Interest
# --------------------------
# In order to ask questions about the behaviour of our individuals with respect
# to the arena, we first need to define the RoIs to represent the separate
# pieces of our arena arena programmatically. Since each part of our arena is
# two-dimensional, we will use ``PolygonOfInterest``s to describe them.
#
# The future `movement plugin for napari <https://github.com/neuroinformatics-unit/movement/pull/393>`_
# will support creating regions of interest by clicking points and drawing
# shapes in the napari GUI. For the time being, we can still define our arena
# by specifying the points that make up the interior and exterior boundaries.
# So first, let's define the boundary vertices of our various regions.

# The centre of the arena is located roughly here
centre = np.array([712.5, 541])
# The "width" (distance between the inner and outer octadecagonal rings) is 40
# pixels wide
ring_width = 40.0
# The distance between opposite vertices of the outer ring is 1090 pixels
ring_extent = 1090.0

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
ring_outer_boundary = ring_extent / 2.0 * unit_shape.copy()
ring_outer_boundary = (
    np.array([ring_outer_boundary.real, ring_outer_boundary.imag]).transpose()
    + centre
)
core_boundary = (ring_extent - ring_width) / 2.0 * unit_shape.copy()
core_boundary = (
    np.array([core_boundary.real, core_boundary.imag]).transpose() + centre
)

nest_corners = ((1245, 585), (1245, 475), (1330, 480), (1330, 580))

# %%
# Our central region is a solid shape without any interior holes.
# To create the appropriate RoI, we just pass the coordinates in either
# clockwise or counter-clockwise order.

central_region = PolygonOfInterest(core_boundary, name="Central region")

# %%
# Likewise, the nest is also just a solid shape without any holes.
# Note that we are only registering the "floor" of the nest here.
nest_region = PolygonOfInterest(nest_corners, name="Nest region")

# %%
# To create an RoI representing the ring region, we need to provide an interior
# boundary so that ``movement`` knows our ring region has a "hole".
# ``movement``s ``PolygonsOfInterest`` can actually support multiple
# (non-overlapping) holes, which is why the ``holes`` argument takes a
# ``list``.
ring_region = PolygonOfInterest(
    ring_outer_boundary, holes=[core_boundary], name="Ring region"
)

arena_fig, arena_ax = plt.subplots(1, 1)
# Overlay an image of the experimental arena
arena_image = sample_data.fetch_dataset_paths(
    "SLEAP_three-mice_Aeon_proofread.analysis.h5"
)["frame"]
arena_ax.imshow(plt.imread(arena_image))

central_region.plot(
    arena_ax, color="lightblue", alpha=0.25, label=central_region.name
)
nest_region.plot(arena_ax, color="green", alpha=0.25, label=nest_region.name)
ring_region.plot(arena_ax, color="blue", alpha=0.25, label=ring_region.name)
arena_ax.legend()
arena_fig.show()

# %%
# View Individual Paths inside the Arena
# --------------------------------------
# We can now overlay the paths that the individuals followed on top of our
# image of the arena and the RoIs that we have defined.

arena_fig, arena_ax = plt.subplots(1, 1)
# Overlay an image of the experimental arena
arena_image = sample_data.fetch_dataset_paths(
    "SLEAP_three-mice_Aeon_proofread.analysis.h5"
)["frame"]
arena_ax.imshow(plt.imread(arena_image))

central_region.plot(
    arena_ax, color="lightblue", alpha=0.25, label=central_region.name
)
nest_region.plot(arena_ax, color="green", alpha=0.25, label=nest_region.name)
ring_region.plot(arena_ax, color="blue", alpha=0.25, label=ring_region.name)

# Plot trajectories of the individuals
mouse_names_and_colours = list(
    zip(positions.individuals.values, ["r", "g", "b"], strict=False)
)
for mouse_name, col in mouse_names_and_colours:
    plot_centroid_trajectory(
        positions,
        individual=mouse_name,
        ax=arena_ax,
        linestyle="-",
        marker=".",
        s=1,
        c=col,
        label=mouse_name,
    )
arena_ax.set_title("Individual trajectories within the arena")
arena_ax.legend()

arena_fig.show()

# %%
# At a glance, it looks like all the individuals remained inside the
# ring-region for the duration of the experiment. We can ask that ``movement``
# check whether this was actually the case, by asking the ``ring_region`` to
# check if the individuals' locations were inside it, at all recorded
# time-points.

# This is a DataArray with dimensions: time, keypoint, and individual.
# The values of the DataArray are True/False values, indicating if at the given
# time, the keypoint of individual was inside ring_region.
individual_was_inside = ring_region.contains_point(positions)
all_individuals_in_ring_at_all_times = individual_was_inside.all()

if all_individuals_in_ring_at_all_times:
    print(
        "All recorded positions, at all times,\n"
        "and for all individuals, were inside ring_region."
    )
else:
    print("At least one position was recorded outside the ring_region.")

# %%
# Compute Distance to Arena Boundary
# ----------------------------------
# Having defined our region of interest, we can begin to ask questions about
# the individual's behaviour during the experiment.
# One thing of particular interest to us might be the distance between an
# individual and the arena walls (boundary). We can query the
# ``PolygonOfInterest`` that we created for this information.

distances_to_boundary = ring_region.compute_distance_to(
    positions, boundary_only=True
)
distances_fig, distances_ax = plt.subplots(1, 1)
for mouse_name, col in mouse_names_and_colours:
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
# Note that we can also obtain the approach vector for each individual, at each
# time-point as well, using the ``compute_approach_vector`` method.
# The distances that we computed above are just the magnitudes of the approach
# vectors that are computed by the following command.

approach_vectors = ring_region.compute_approach_vector(
    positions, boundary_only=True
)

# %%
# That ``compute_distance_to`` is returning the distance to the closest
# point on the arena's boundary, which could be either the interior or exterior
# wall. If we wanted to explicitly know the distance to the exterior wall, for
# example, we can call this on the corresponding attribute of our
# ``PolygonOfInterest``.

distances_to_exterior = ring_region.exterior_boundary.compute_distance_to(
    positions
)
distances_exterior_fig, distances_exterior_ax = plt.subplots(1, 1)
for mouse_name, col in mouse_names_and_colours:
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
# The ``boundary_only`` Keyword
# -----------------------------
# The ``boundary_only`` argument that is passed to ``compute_distance_to``
# should be noted. By passing this argument as ``True``, we force ``movement``
# to compute the distance to the boundary of our region of interest. If we
# passed this value as ``False``, then ``movement`` would treat points inside
# the region as being a distance 0 from the region - and so the resulting
# distances would all be 0!

distances_to_region = ring_region.compute_distance_to(
    positions, boundary_only=False
)

print(
    "distances_to_region are all zero:", np.allclose(distances_to_region, 0.0)
)

# %%
# The ``boundary_only`` keyword argument can be provided to a number of methods
# of the ``PolygonOfInterest`` and ``LineOfInterest`` classes. In each case,
# the effect is the same; to toggle whether interior points of the region are
# considered part of the region (``boundary_only = False``) or not
# (``boundary_only = True``). Care should be taken when dealing with 1D regions
# of interest like segments or dividing walls, since the "boundary" of a 1D
# object is considered to be just the endpoints!

# %%
# Boundary Angles
# ---------------
# Having observed the individuals' behaviour, we can begin to ask questions
# about their orientation with respect to the boundary. ``movement`` currently
# supports the computation of two such "boundary angle"s;
#
# - The allocentric boundary angle. Given a region of interest :math:`R`,
#   reference vector :math:`\vec{r}` (such as global north), and a position
#   :math:`p` in space (e.g. the position of an individual), the allocentric
#   boundary angle is the signed angle between the approach vector from
#   :math:`p` to :math:`R`, and :math:`\vec{r}`.
# - The egocentric boundary angle. Given a region of interest :math:`R`, a
#   forward vector :math:`\vec{f}` (e.g. the direction of travel of an
#   individual), and the point of origin of :math:`\vec{f}` denoted :math:`p`
#   (e.g. the individual's current position), the egocentric boundary angle is
#   the signed angle between the approach vector from :math:`p` to :math:`R`,
#   and :math:`\vec{f}`.
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

inner_wall = ring_region.interior_boundaries[0]
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

for mouse_name, col in mouse_names_and_colours:
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
# of the angle sign indicates that the individual undertook a sudden U-turn.
# That is, switched from travelling clockwise around the enclosure to
# anticlockwise (or vice-versa).
