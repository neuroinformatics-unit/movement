"""Compute and visualise kinematics
====================================

Compute displacement, velocity and acceleration, and
visualise the results.
"""

# %%
# Imports
# -------

# For interactive plots: install ipympl with `pip install ipympl` and uncomment
# the following line in your notebook
# %matplotlib widget
import numpy as np
from matplotlib import pyplot as plt

import movement.kinematics as kin
from movement import sample_data
from movement.plots import plot_centroid_trajectory
from movement.utils.vector import compute_norm

# %%
# Load sample dataset
# ------------------------
# First, we load an example dataset. In this case, we select the
# ``SLEAP_three-mice_Aeon_proofread`` sample data.
ds = sample_data.fetch_dataset(
    "SLEAP_three-mice_Aeon_proofread.analysis.h5",
)

print(ds)

# %%
# We can see in the printed description of the dataset ``ds`` that
# the data was acquired at 50 fps, and the time axis is expressed in seconds.
# It includes data for three individuals(``AEON3B_NTP``, ``AEON3B_TP1``,
# and ``AEON3B_TP2``), and only one keypoint called ``centroid`` was tracked
# in ``x`` and ``y`` dimensions.

# %%
# The loaded dataset ``ds`` contains two data arrays:
# ``position`` and ``confidence``.
# To compute displacement, velocity and acceleration, we will need the
# ``position`` one:
position = ds.position


# %%
# Visualise the data
# ---------------------------
# First, let's visualise the trajectories of the mice in the XY plane,
# colouring them by individual.
# For this we can use :func:`movement.plots.plot_centroid_trajectory`
# which is a wrapper around :func:`matplotlib.pyplot.scatter`.
# The fig and ax objects returned can be used to further customise the plot.

# Create a single figure and axes
fig, ax = plt.subplots(1, 1)
# Invert y-axis so (0,0) is in the top-left,
# matching typical image coordinate systems
ax.invert_yaxis()
# Plot trajectories for each mouse on the same axes
for mouse_name, col in zip(
    position.individuals.values,
    ["r", "g", "b"],  # colours
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
ax.set_xlabel("x (pixels)")
ax.set_ylabel("y (pixels)")
fig.show()

# %%
# We can see that the trajectories of the three mice are close to a circular
# arc. Notice that the x and y axes are set to equal scales, and that the
# origin of the coordinate system is at the top left of the image. This
# follows the convention for SLEAP and most image processing tools.

# %%
# We can also plot the trajectories of the mice in the XY plane independently,
# colouring the data points based on their timestamps. This is the default
# behaviour of
# :func:`plot_centroid_trajectory()<movement.plots.plot_centroid_trajectory>`
# when the ``c`` argument is not provided:
fig, axes = plt.subplots(2, 2, sharey=True)
for mouse_name, ax in zip(
    position.individuals.values, axes.flat, strict=False
):
    ax.invert_yaxis()
    fig, ax = plot_centroid_trajectory(
        position,
        individual=mouse_name,
        ax=ax,
        s=2,
    )
    ax.set_aspect("equal")
    ax.set_xlim(150, 1250)
    ax.set_ylim(500, 1100)
    ax.set_title(f"Trajectory {mouse_name}")
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.collections[0].colorbar.set_label("Time (frames)")
# Hide the unused subplot (4th one)
axes[1, 1].set_visible(False)
fig.tight_layout()
fig.show()

# %%
# These plots show that for this snippet of the data,
# two of the mice (``AEON3B_NTP`` and ``AEON3B_TP1``)
# moved around the circle in clockwise direction, and
# the third mouse (``AEON3B_TP2``) followed an anti-clockwise direction.

# %%
# We can also inspect the components of the position vector against time
# using ``xarray``'s built-in plotting methods. We use
# :meth:`xarray.DataArray.squeeze` to
# remove the dimension of length 1 from the data (the ``keypoints`` dimension).
position.squeeze().plot.line(x="time", row="individuals", aspect=2, size=2.5)
plt.gcf().show()

# %%
# If we use ``xarray``'s plotting function, the axes units are automatically
# taken from the data array. In our case, ``time`` is expressed in seconds,
# and the ``x`` and ``y`` coordinates of the ``position`` are in pixels.

# %%
# Compute displacement vectors
# ----------------------------
# The :mod:`movement.kinematics` module
# provides functions to compute various kinematic variables,
# such as displacement, velocity, and acceleration. Below we showcase
# how these functions can be used.
#
# We can compute the forward displacement vectors as follows:
forward_displacement = kin.compute_forward_displacement(position)

# %%
# The :func:`movement.kinematics.compute_forward_displacement`
# function will return a data array equivalent to the ``position`` one,
# but holding displacement data along the ``space`` axis.
#
# The ``forward_displacement`` data array holds, for a given individual and
# keypoint at timestep ``t``, the vector that goes from its current position
# at time ``t`` to its next position at time ``t+1``.

# %%
# And what happens in the last timestep, when there is no next timepoint?
# We define the forward displacement vector then to be the
# zero vector. This way the shape of the ``forward_displacement`` data array
# is the same as the ``position`` array:
print(f"Shape of position: {position.shape}")
print(f"Shape of displacement: {forward_displacement.shape}")

# %%
# We can visualise the forward displacement vectors with a quiver plot. In
# this case we focus on the mouse ``AEON3B_TP2``:
mouse_name = "AEON3B_TP2"

fig = plt.figure()
ax = fig.add_subplot()

# plot position data
sc = ax.scatter(
    position.sel(individuals=mouse_name, space="x"),
    position.sel(individuals=mouse_name, space="y"),
    s=15,
    c=position.time,
    cmap="viridis",
)

# plot forward displacement vectors: at t, vector from t to t+1
ax.quiver(
    position.sel(individuals=mouse_name, space="x"),
    position.sel(individuals=mouse_name, space="y"),
    forward_displacement.sel(individuals=mouse_name, space="x"),
    forward_displacement.sel(individuals=mouse_name, space="y"),
    angles="xy",
    scale=1,
    scale_units="xy",
    headwidth=7,
    headlength=9,
    headaxislength=9,
)

ax.set_xlim(480, 600)
ax.set_ylim(980, 1080)
ax.set_xlabel("x (pixels)")
ax.set_ylabel("y (pixels)")
ax.set_title(f"Zoomed in forward trajectory of {mouse_name}")
ax.invert_yaxis()
sc.set_clim(8.8, 9.2)
fig.colorbar(sc, ax=ax, label="time (s)")


# %%
# We can visually verify that indeed the forward displacement vector
# connects the previous and current positions as expected.

# %%
# Similarly, with :func:`movement.kinematics.compute_backward_displacement`
# we can compute the backward displacement vectors, which connect the current
# position to the previous one:
backward_displacement = kin.compute_backward_displacement(position)

# %%
# In this case, the backward displacement vector at the first timestep
# is defined as the zero vector, since there is no previous position.

# %%
# Adapting the code snippet from above, we can visually check that the
# backward displacement vector is indeed the reverse of the forward
# displacement vector.

fig = plt.figure()
ax = fig.add_subplot()

sc = ax.scatter(
    position.sel(individuals=mouse_name, space="x"),
    position.sel(individuals=mouse_name, space="y"),
    s=15,
    c=position.time,
    cmap="viridis",
)

ax.quiver(
    position.sel(individuals=mouse_name, space="x"),
    position.sel(individuals=mouse_name, space="y"),
    backward_displacement.sel(individuals=mouse_name, space="x"),
    backward_displacement.sel(individuals=mouse_name, space="y"),
    angles="xy",
    scale=1,
    scale_units="xy",
    headwidth=7,
    headlength=9,
    headaxislength=9,
)

ax.set_xlim(480, 600)
ax.set_ylim(980, 1080)
ax.set_xlabel("x (pixels)")
ax.set_ylabel("y (pixels)")
ax.set_title(f"Zoomed in backward trajectory of {mouse_name}")
ax.invert_yaxis()
sc.set_clim(8.8, 9.2)
fig.colorbar(sc, ax=ax, label="time (s)")


# %%
# Compute path length
# --------------------
# We can compute the distance travelled by the
# mouse as the sum of the lengths of all
# displacement vectors along its trajectory.
# Both backward and forward displacement vectors
# should give the same result:

# length of each forward displacement vector
forward_displacement_lengths = compute_norm(
    forward_displacement.sel(individuals=mouse_name)
)

# length of each backward displacement vector
backward_displacement_lengths = compute_norm(
    backward_displacement.sel(individuals=mouse_name)
)

# check their lengths are the same
np.testing.assert_almost_equal(
    forward_displacement_lengths.values[:-1],  # exclude last timestep
    backward_displacement_lengths.values[1:],  # exclude first timestep
)

# sum the lengths of all displacement vectors (in pixels)
total_displacement_fwd = forward_displacement_lengths.sum(dim="time").values[0]
total_displacement_bwd = backward_displacement_lengths.sum(dim="time").values[
    0
]

print(
    f"The mouse {mouse_name}'s path length is {total_displacement_fwd:.2f} "
    "pixels long (using forward displacement)"
)
print(
    f"The mouse {mouse_name}'s path length is {total_displacement_bwd:.2f} "
    "pixels long (using backward displacement)"
)


# %%
# We provide a convenience function
# :func:`movement.kinematics.compute_path_length`
# to compute the path length for all individuals and keypoints in a position
# data array. We can verify that using this function gives the same result
# as before for the ``AEON3B_TP2`` mouse:

path_lengths = kin.compute_path_length(ds.position)

for mouse_name in path_lengths.individuals.values:
    print(
        f"Path length for {mouse_name}: "
        f"{path_lengths.sel(individuals=mouse_name).values[0]:.2f} pixels"
    )

# %%
# Compute velocity
# ----------------
# We can also compute the velocity vectors for all individuals in our data
# array:
velocity = kin.compute_velocity(position)

# %%
# The :func:`movement.kinematics.compute_velocity`
# function will return a data array equivalent to
# the ``position`` one, but holding velocity data along the ``space`` axis,
# rather than position data. Notice how ``xarray`` nicely deals with the
# different individuals and spatial dimensions for us! ✨

# %%
# We can plot the components of the velocity vector against time
# using ``xarray``'s built-in plotting methods. We use
# :meth:`xarray.DataArray.squeeze` to
# remove the dimension of length 1 from the data (the ``keypoints`` dimension).

velocity.squeeze().plot.line(x="time", row="individuals", aspect=2, size=2.5)
plt.gcf().show()

# %%
# The components of the velocity vector seem noisier than the components of
# the position vector.
# This is expected, since we are estimating the velocity using differences in
# position (which is somewhat noisy), over small stepsizes.
# More specifically, we use :func:`numpy.gradient` internally, which
# uses second order central differences.

# %%
# We can also visualise the speed, as the magnitude (norm)
# of the velocity vector:
fig, axes = plt.subplots(3, 1, sharex=True, sharey=True)
for mouse_name, ax in zip(velocity.individuals.values, axes, strict=False):
    # compute the magnitude of the velocity vector for one mouse
    speed_one_mouse = compute_norm(velocity.sel(individuals=mouse_name))
    # plot speed against time
    ax.plot(speed_one_mouse)
    ax.set_title(mouse_name)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("speed (px/s)")
fig.tight_layout()

# %%
# To visualise the direction of the velocity vector at each timestep, we can
# again use a quiver plot:
mouse_name = "AEON3B_TP2"
fig = plt.figure()
ax = fig.add_subplot()
# plot trajectory (position data)
sc = ax.scatter(
    position.sel(individuals=mouse_name, space="x"),
    position.sel(individuals=mouse_name, space="y"),
    s=15,
    c=position.time,
    cmap="viridis",
)
# plot velocity vectors
ax.quiver(
    position.sel(individuals=mouse_name, space="x"),
    position.sel(individuals=mouse_name, space="y"),
    velocity.sel(individuals=mouse_name, space="x"),
    velocity.sel(individuals=mouse_name, space="y"),
    angles="xy",
    scale=2,
    scale_units="xy",
    color="r",
)
ax.axis("equal")
ax.set_xlabel("x (pixels)")
ax.set_ylabel("y (pixels)")
ax.set_title(f"Velocity quiver plot for {mouse_name}")
ax.invert_yaxis()
fig.colorbar(sc, ax=ax, label="time (s)")
fig.show()

# %%
# Here we scaled the length of vectors to half of their actual value
# (``scale=2``) for easier visualisation.

# %%
# Compute acceleration
# ---------------------
# Let's now compute the acceleration for all individuals in our data
# array:
accel = kin.compute_acceleration(position)

# %%
# and plot of the components of the acceleration vector ``ax``, ``ay`` per
# individual:
fig, axes = plt.subplots(3, 1, sharex=True, sharey=True)
for mouse_name, ax in zip(accel.individuals.values, axes, strict=False):
    # plot x-component of acceleration vector
    ax.plot(
        accel.sel(individuals=mouse_name, space=["x"]).squeeze(),
        label="ax",
    )
    # plot y-component of acceleration vector
    ax.plot(
        accel.sel(individuals=mouse_name, space=["y"]).squeeze(),
        label="ay",
    )
    ax.set_title(mouse_name)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("speed (px/s**2)")
    ax.legend(loc="center right", bbox_to_anchor=(1.07, 1.07))
fig.tight_layout()

# %%
# We can also compute and visualise the magnitude (norm) of the
# acceleration vector for each individual:
fig, axes = plt.subplots(3, 1, sharex=True, sharey=True)
for mouse_name, ax in zip(accel.individuals.values, axes, strict=False):
    # compute magnitude of the acceleration vector for one mouse
    accel_one_mouse = compute_norm(accel.sel(individuals=mouse_name))

    # plot acceleration against time
    ax.plot(accel_one_mouse)
    ax.set_title(mouse_name)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("accel (px/s**2)")
fig.tight_layout()
