"""
Compute and visualise kinematics
============================

Compute displacement, velocity and acceleration data on an example dataset
and visualise the results.
"""

# %%
# Imports
# -------

import numpy as np

# For interactive plots: install ipympl with `pip install ipympl` and uncomment
# the following line in your notebook
# %matplotlib widget
from matplotlib import pyplot as plt

import movement.analysis.kinematics as kin
from movement import sample_data

# %%
# Load sample dataset
# ------------------------
# First, we load an example dataset. In this case, we select the
# `SLEAP_three-mice_Aeon_proofread` sample data.
ds = sample_data.fetch_sample_data(
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
# ``pose_tracks`` and ``confidence``.
# To compute displacement, velocity and acceleration, we will need the pose
# tracks one:
pose_tracks = ds.pose_tracks


# %%
# Visualise the data
# ---------------------------
# First, let's visualise the trajectories of the mice in the XY plane,
# colouring them by individual.

fig, ax = plt.subplots(1, 1)
for mouse_name, col in zip(pose_tracks.individuals.values, ["r", "g", "b"]):
    ax.plot(
        pose_tracks.sel(individuals=mouse_name, space="x"),
        pose_tracks.sel(individuals=mouse_name, space="y"),
        linestyle="-",
        marker=".",
        markersize=2,
        linewidth=0.5,
        c=col,
        label=mouse_name,
    )
    ax.invert_yaxis()
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.axis("equal")
    ax.legend()

# %%
# We can see that the trajectories of the three mice are close to a circular
# arc. Notice that the x and y axes are set to equal scales, and that the
# origin of the coordinate system is at the top left of the image. This
# follows the convention for SLEAP and most image processing tools.

# %%
# We can also color the data points based on their timestamps:
fig, axes = plt.subplots(3, 1, sharey=True)
for mouse_name, ax in zip(pose_tracks.individuals.values, axes):
    sc = ax.scatter(
        pose_tracks.sel(individuals=mouse_name, space="x"),
        pose_tracks.sel(individuals=mouse_name, space="y"),
        s=2,
        c=pose_tracks.time,
        cmap="viridis",
    )

    ax.set_title(mouse_name)
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.axis("equal")
    fig.colorbar(sc, ax=ax, label="time (s)")
fig.tight_layout()

# %%
# These plots show that for this snippet of the data,
# two of the mice (``AEON3B_NTP`` and ``AEON3B_TP1``)
# moved around the circle in anti-clockwise direction, and
# the third mouse (``AEON3B_TP2``) followed a clockwise direction.

# %%
# We can also easily plot the components of the position vector against time
# using ``xarray``'s built-in plotting methods. We use ``squeeze()`` to
# remove the dimension of length 1 from the data (the keypoints dimension).
pose_tracks.squeeze().plot.line(
    x="time", row="individuals", aspect=2, size=2.5
)
plt.gcf().show()

# %%
# The axes units are automatically taken from the data array. In our case,
# ``time`` is expressed in seconds,
# and the ``x`` and ``y`` coordinates of the ``pose_tracks`` are in pixels.

# %%
# Compute displacement
# ---------------------
# We can start off by computing the distance travelled by the mice along
# their trajectories.
# For this, we can use the ``displacement`` method of the ``move`` accessor.
displacement = ds.move.displacement

# %%
# Notice that we could also compute the displacement (and all the other
# kinematic variables) using the kinematics module:

# %%
displacement_kin = kin.compute_displacement(pose_tracks)

# %%
# However, we encourage our users to familiarise themselves with the more
# convenient syntax of the ``move`` accessor.

# %%
# The ``displacement``` data array holds, for a given individual, and keypoint
# at timestep ``t``, the vector from its previous position (``t-1``) to its
# current one (``t``).
# %%
# We define the displacement vector at time ``t=0`` to be the zero vector.
# This way the shape of the ``pose_tracks_displacement`` data array is the
# same as the  ``pose_tracks`` array:
print(f"Shape of pose_tracks: {pose_tracks.shape}")
print(f"Shape of pose_tracks_displacement: {displacement.shape}")

# %%
# We can visualise these displacement vectors with a quiver plot. In this case
# we focus on the mouse ``AEON3B_TP2``:
mouse_name = "AEON3B_TP2"

fig = plt.figure()
ax = fig.add_subplot()

# plot position data
sc = ax.scatter(
    pose_tracks.sel(individuals=mouse_name, space="x"),
    pose_tracks.sel(individuals=mouse_name, space="y"),
    s=15,
    c=pose_tracks.time,
    cmap="viridis",
)

# plot displacement vectors
# the displacement vector at t is plotted:
# - with its origin at position(t)
# - with its tip at position (t-1)
ax.quiver(
    pose_tracks.sel(individuals=mouse_name, space="x"),
    pose_tracks.sel(individuals=mouse_name, space="y"),
    -displacement.sel(individuals=mouse_name, space="x"),
    -displacement.sel(individuals=mouse_name, space="y"),
    angles="xy",
    scale=1,
    scale_units="xy",
    headwidth=7,
    headlength=9,
    headaxislength=9,
)
ax.axis("equal")
ax.set_xlim(450, 575)
ax.set_ylim(950, 1075)
ax.set_xlabel("x (pixels)")
ax.set_ylabel("y (pixels)")
ax.set_title(f"Zoomed in trajectory of {mouse_name}")
fig.colorbar(sc, ax=ax, label="time (s)")

# %%
# Notice that we invert the sign of the displacement vector in the plot for an
# easier visual check. The displacement vector is defined as the vector at
# time ``t``, that goes from the previous position point ``t-1`` to the
# current position point at ``t``. Therefore, the opposite vector will point
# from the position point at ``t``, to the position point at ``t-1``.
# This opposite vector is what we represent in the plot.

# %%
# With the displacement data we can compute the distance travelled by the
# mouse along the curve:
displacement_vectors_lengths = np.linalg.norm(
    displacement.sel(individuals=mouse_name, space=["x", "y"]).squeeze(),
    axis=1,
)

total_displacement = np.sum(displacement_vectors_lengths, axis=0)  # pixels

print(
    f"The mouse {mouse_name}'s trajectory is {total_displacement:.2f} "
    "pixels long"
)

# %%
# Compute velocity
# ----------------
# We can easily compute the velocity vectors for all individuals in our data
# array:
velocity = ds.move.velocity

# %%
# ``xarray`` nicely deals with the different individuals and spatial
# dimensions for us! ✨

# %%
# We can plot the components of the velocity vector against time
# using ``xarray``'s built-in plotting methods. We use ``squeeze()`` to
# remove the dimension of length 1 from the data (the keypoints dimension).

velocity.squeeze().plot.line(x="time", row="individuals", aspect=2, size=2.5)
plt.gcf().show()

# %%
# The components of the velocity vector seem noisier than the components of
# the position vector.
# This is expected, since we are deriving the velocity using differences in
# position (which is somewhat noisy), over small stepsizes.
# More specifically, we use numpy's gradient implementation, which
# uses second order accurate central differences.

# %%
# We can also visualise the speed, as the norm of the velocity vector:
fig, axes = plt.subplots(3, 1, sharex=True, sharey=True)
for mouse_name, ax in zip(velocity.individuals.values, axes):
    # compute the norm of the velocity vector for a mouse
    speed_one_mouse = np.linalg.norm(
        velocity.sel(individuals=mouse_name, space=["x", "y"]).squeeze(),
        axis=1,
    )
    ax.plot(speed_one_mouse)
    ax.set_title(mouse_name)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("speed (px/s)")
fig.tight_layout()

# %%
# To visualise the direction of the velocity vector at each timestep, we can
# use a quiver plot:
mouse_name = "AEON3B_TP2"
fig = plt.figure()
ax = fig.add_subplot()
sc = ax.scatter(
    pose_tracks.sel(individuals=mouse_name, space="x"),
    pose_tracks.sel(individuals=mouse_name, space="y"),
    s=15,
    c=pose_tracks.time,
    cmap="viridis",
)
ax.quiver(
    pose_tracks.sel(individuals=mouse_name, space="x"),
    pose_tracks.sel(individuals=mouse_name, space="y"),
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
fig.colorbar(sc, ax=ax, label="time (s)")
fig.show()

# %%
# Here we scaled the length of vectors to half of their actual value
# (``scale=2``) for easier visualisation.

# %%
# Compute acceleration
# ---------------------
# We can compute the accelaration of the data with an equivalent method:
accel = ds.move.acceleration

# %%
# and plot of the components of the acceleration vector (`ax`, `ay`) per
# individual:
fig, axes = plt.subplots(3, 1, sharex=True, sharey=True)
for mouse_name, ax in zip(accel.individuals.values, axes):
    ax.plot(
        accel.sel(individuals=mouse_name, space=["x"]).squeeze(),
        label="ax",
    )
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
# The norm of the acceleration vector would give us the magnitude of the
# acceleration.
# We can also represent this for each individual.
fig, axes = plt.subplots(3, 1, sharex=True, sharey=True)
for mouse_name, ax in zip(accel.individuals.values, axes):
    accel_one_mouse = np.linalg.norm(
        accel.sel(individuals=mouse_name, space=["x", "y"]).squeeze(),
        axis=1,
    )
    ax.plot(accel_one_mouse)
    ax.set_title(mouse_name)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("accel (px/s**2)")
fig.tight_layout()

# %%
