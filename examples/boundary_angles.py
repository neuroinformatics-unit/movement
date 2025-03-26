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
# In this example, we will look at how we can use the functionality for regions
# of interest (RoIs) provided by ``movement`` to analyse our sample dataset.

# %%
# Define Regions of Interest
# --------------------------
# In order to ask questions about the behaviour of our individuals with respect
# to the arena, we first need to define the RoIs to represent the separate
# pieces of our arena arena programmatically. Since each part of our arena is
# two-dimensional, we will use a ``PolygonOfInterest`` to describe each of
# them.
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
# Compute Distance to the Nest
# ----------------------------
# Defining RoIs means that we can efficiently extract information from our data
# that depends on the location or relative position of an individual to an RoI.
# For example, we might be interested in how the distance between an
# individual and the nest region changes over the course of the experiment. We
# can query the ``nest_region`` that we created for this information.

# Compute all distances to the nest; for all times, keypoints, and
# individuals.
distances_to_nest = nest_region.compute_distance_to(positions)
distances_fig, distances_ax = plt.subplots(1, 1)
for mouse_name, col in mouse_names_and_colours:
    distances_ax.plot(
        distances_to_nest.sel(individuals=mouse_name),
        c=col,
        label=mouse_name,
    )
distances_ax.legend()
distances_ax.set_xlabel("Time (frames)")
distances_ax.set_ylabel("Distance to nest_region (pixels)")
distances_fig.show()

# %%
# We can see that the ``AEON38_TP2`` individual appears to be moving away from
# the nest during the experiment, whilst the other two individuals are
# approaching the nest. The "plateau" in the figure between frames 200-400 is
# when the individuals meet in the ring_region, and remain largely stationary
# in a group until they can pass each other.
#
# One other thing to note is that ``compute_distance_to`` is returning the
# distance "as the crow flies" to the ``nest_region``. This means that
# structures potentially in the way (such as the ``ring_region`` walls) are not
# accounted for in this distance calculation. Further to this, the "distance to
# a RoI" should always be understood as "the distance from a point to the
# closest point within an RoI".
#
# If we wanted to check the direction of closest approach to a region, referred
# to as the **approach vector**, we can use the ``compute_approach_vector``
# method.
# The distances that we computed via ``compute_distance_to`` are just the
# magnitudes of the approach vectors.

approach_vectors = nest_region.compute_approach_vector(positions)

# %%
# The ``boundary_only`` Keyword
# -----------------------------
# We know that our individuals spend the duration of the experiment within the
# ``ring_region`. From our plot of the distances to the nest, we saw that there
# appears to be a time-window in which the individuals are grouped up, possibly
# trying to pass each other as they approach from different directions.
# We might want to see if this is the case by examining the distance between
# each individual and the boundary of the ``ring_region`` - if we expect the
# individuals to move to opposite sides of the ``ring_region`` in order to pass
# each other.
# However, if we try the same commands as before, we will get something
# slightly unexpected:

distances_to_ring_wall = ring_region.compute_distance_to(positions)
distances_fig, distances_ax = plt.subplots(1, 1)
for mouse_name, col in mouse_names_and_colours:
    distances_ax.plot(
        distances_to_ring_wall.sel(individuals=mouse_name),
        c=col,
        label=mouse_name,
    )
distances_ax.legend()
distances_ax.set_xlabel("Time (frames)")
distances_ax.set_ylabel("Distance to closest ring_region wall (pixels)")

print(
    "distances_to_ring_wall are all zero:",
    np.allclose(distances_to_ring_wall, 0.0),
)

distances_fig.show()

# %%
# So why is every distance returning zero? This is because of the convention
# for finding the nearest point in a region, that was mentioned earlier. The
# distance of a point ``p`` to an RoI is defined as the distance from ``p`` to
# the closest point within the RoI. However, if ``p`` is inside the RoI - like
# we have here - then the closest point within the RoI is ``p`` itself!
#
# In cases such as this, what we really want is the distance to the closest
# point *on the boundary* of ``ring_region``. In anticipation of this
# requirement, most RoI methods accept the ``boundary_only`` keyword argument.
# In each case, the effect of passing ``boundary_only`` is the same; it toggles
# whether interior points of an RoI should be considered part of the region
# (``boundary_only = False``) or not (``boundary_only = True``). Care should be
# taken when dealing with 1D regions of interest like segments or dividing
# walls, since the "boundary" of a 1D object is considered to be just the
# endpoints!
#
# Armed with this knowledge, we can fix the analysis we just tried to run.

distances_to_ring_wall = ring_region.compute_distance_to(
    positions, boundary_only=True
)
distances_fig, distances_ax = plt.subplots(1, 1)
for mouse_name, col in mouse_names_and_colours:
    distances_ax.plot(
        distances_to_ring_wall.sel(individuals=mouse_name),
        c=col,
        label=mouse_name,
    )
distances_ax.legend()
distances_ax.set_xlabel("Time (frames)")
distances_ax.set_ylabel("Distance to closest ring_region wall (pixels)")

print(
    "distances_to_ring_wall are all zero:",
    np.allclose(distances_to_ring_wall, 0.0),
)

distances_fig.show()

# %%
# The resulting plot looks much more like what we expect, in that we see some
# variation in the distance to the ``ring_region`` walls now.
# However, this again is not very helpful; ``compute_distance_to`` is returning
# the distance to the closest point on ``ring_region``'s boundary, which could
# be _either_ the interior or exterior wall. This means that we won't be able
# to tell if the individuals do move to opposite walls to pass one another.
# Instead, let's ask for the distance to the exterior wall, rather than just
# the closest wall.

# Note that the exterior_boundary of the ring_region is a 1D RoI (a series of
# connected line segments). As such, boundary_only needs to be False.
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
# This output is much more helpful. We see that the individuals are largely the
# same distance from the exterior wall during frames 250-350, and then notice
# that;
#
# - Individual ``AEON_TP1`` moves far away from the exterior wall,
# - ``AEON3B_NTP`` moves almost up to the exterior wall,
# - and ``AEON3B_TP2`` seems to remain between the other two individuals.
#
# After frame 400, the individuals appear to go back to chaotic distances from
# the exterior wall again, which is consistent with them having passed each
# other in the ``ring_region`` and once again having the entire width of the
# ring to explore.

# %%
# Boundary Angles
# ---------------
# Having observed the individuals' behaviour as they pass one another in the
# ``ring_region``, we can begin to ask questions about their orientation with
# respect to the nest. ``movement`` currently supports the computation of two
# such "boundary angle"s;
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
# Note that egocentric angles are generally computed with changing frames of
# reference in mind - the forward vector may be varying in time as the
# individual moves around the arena. By contrast, allocentric angles are always
# computed with respect to some fixed reference frame.
#
# For the purposes of our example, we will define our "forward vector" as the
# displacement vector between successive time-points, for each individual.
# We will also define our reference frame, or "global north" direction to be
# the direction of the positive x-axis.

forward_vector = positions.diff(dim="time", label="lower")
global_north = np.array([1.0, 0.0])

allocentric_angles = nest_region.compute_allocentric_angle_to_nearest_point(
    positions,
    reference_vector=global_north,
    in_degrees=True,
)
egocentric_angles = nest_region.compute_egocentric_angle_to_nearest_point(
    forward_vector,
    positions[:-1],
    in_degrees=True,
)

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
allo_ax.set_ylabel("Allocentric angle (degrees)")
allo_ax.legend()

angle_plot.tight_layout()
angle_plot.show()

# %%
# We observe that the allocentric angles display a step-like behaviour. This
# is because the allocentric angle is computed solely from an individual's
# position relative to a RoI, and does not care about a forward vector. As
# such, we see a similar shape for the graph of the allocentric angle as we did
# for the distance to the nest (though inverted in the y-axis), with the same
# "plateau" during the time window where the individuals are trying to pass one
# another (hence their positions are largely constant until they find a way
# passed one another).
#
# By contrast, the egocentric angles display many large fluctuations, but
# (excluding the time between frames 200-400 where the individuals are trying
# to pass one another) display a slight increasing or decreasing trend. These
# general trends are to be expected; individual ``AEON3B_TP2`` is moving
# clockwise around the ``ring_region``, thus the angle between its forward
# vector and the approach vector to the nest is getting shallower as it
# continues on its path. Conversely, the other two individuals are moving
# counter-clockwise around the ``ring_region``, so their egocentric angles are
# gradually increasing.
#
# The time interval in which the individuals are attempting to pass each other
# shows large fluctuations in egocentric angle, because the forward vectors of
# the individuals are rapidly changing direction as they attempt to move out of
# each other's way. The egocentric angle is sensitive to changes in the forward
# vector, unlike the allocentric angle (which remains largely constant during
# this time).
