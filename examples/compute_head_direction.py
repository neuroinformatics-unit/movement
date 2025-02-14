"""Compute head direction
=========================

Various ways to compute the head direction vector and angle.
"""
# %%
# Imports
# -------

# For interactive plots: install ipympl with `pip install ipympl` and uncomment
# the following line in your notebook
# %matplotlib widget
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt

from movement import sample_data
from movement.io import load_poses
from movement.kinematics import (
    compute_forward_vector,
)
from movement.plots import plot_trajectory
from movement.utils.vector import cart2pol, pol2cart

xr.set_options(keep_attrs=True)

# %%
# Load sample dataset
# ------------------------
# In this tutorial, we will use a sample dataset with a single individual
# (a mouse) and six keypoints.

dataset_name = "SLEAP_single-mouse_EPM.analysis.h5"
ds_path = sample_data.fetch_dataset_paths(dataset_name)["poses"]
ds = load_poses.from_sleap_file(ds_path, fps=None)  # force time_unit = frames

print(ds)
print("-----------------------------")
print(f"Individuals: {ds.individuals.values}")
print(f"Keypoints: {ds.keypoints.values}")

# %%
# The loaded dataset ``ds`` contains two data variables:``position`` and
# ``confidence``. Both are stored as data arrays. In this tutorial, we will
# use only ``position``:
position = ds.position


# %%
# Motivation
# ----------
# The head direction of an animal can be an important feature in behavioural
# analyses, for example to deduce where the animal is looking or to infer its
# focus of attention.
#
# In this notebook, we will demonstrate several ways
# to compute the head direction—which can be represented as a vector
# or an angle—based on the position of several keypoints.

# %%
# Visualise the head trajectory
# -----------------------------
# We can start by visualising the head trajectory, taking the midpoint
# between the two ears as a proxy for the head position.
# We will overlay that on a single video frame that comes
# as part of the sample dataset.
#
# The :func:`plot_trajectory()<movement.plots.trajectory.plot_trajectory>`
# function can help you visualise the trajectory of any keypoint in the data.
# Passing a list of keypoints, in this case ``["left_ear", "right_ear"]``,
# will plot the centroid (midpoint) of the selected keypoints.
# By default, the first individual in the dataset is shown.

# Read sample video frame
frame_path = sample_data.fetch_dataset_paths(dataset_name)["frame"]

# Create figure and axis
fig, ax = plt.subplots(1, 1)

# Plot the frame using imshow
frame = plt.imread(frame_path)
ax.imshow(frame)

# Plot the trajectory of ears midpoint on the same axis
plot_trajectory(
    ds.position,
    keypoints=["left_ear", "right_ear"],
    ax=ax,
    # arguments forwarded to plt.scatter
    s=10,
    cmap="viridis",
    marker="o",
    alpha=0.05,
)

# Adjust title
ax.set_title("Head trajectory")
fig.show()

# %%
# We can see that the majority of the head trajectory data is within a
# cruciform shape, because the mouse is moving on an
# `Elevated Plus Maze <https://en.wikipedia.org/wiki/Elevated_plus_maze>`_.
#
# The plot suggests the mouse spends most of its time in the
# covered arms of the maze.

# %%
# Compute the head-to-snout vector
# --------------------------------
# We can choose to define head direction as the vector from the middle
# of the head (midpoint between ears) to the front of the head (the snout).

# compute the position of midpoint between the ears
midpoint_ears = position.sel(keypoints=["left_ear", "right_ear"]).mean(
    dim="keypoints"
)
# snout position
# (`drop=True` removes the keypoints dimension, which is now redundant)
snout = position.sel(keypoints="snout", drop=True)

# compute the head vector as the difference between the snout and the
# midpoint between the ears.
head_vector = position.sel(keypoints="snout") - midpoint_ears

# %%
# .. note::
#   You can think of each point's position as a 2D vector, with its base at the
#   origin (for image coordinates, that's the center of the pixel at
#   the top-left corner of the image) and its tip at the point's position.
#
#   The vector that goes form point :math:`U` to point :math:`V` can be
#   computed as the difference :math:`\vec{v} - \vec{u}`, i.e.
#   "tip - base" (see the image below).
#
#   .. image:: ../_static/Vector-Subtraction.png
#     :width: 600
#     :alt: Schematic showing vector subtraction


# %%
# 2D vectors in polar coordinates
# -------------------------------
# A convenient way to inspect the orientation angle of a vector in 2D is by
# expressing it in polar coordinates. We can do this with the vector function
# ``cart2pol``:

head_vector_polar = cart2pol(head_vector)

print(head_vector_polar)

# %%
# Notice how the resulting array has a ``space_pol`` dimension with two
# coordinates: ``rho`` and ``phi``. These are the polar coordinates of the
# head vector.
#
# The coordinate ``rho`` is the norm (i.e., magnitude, length) of the vector.
# In our case, the distance from the midpoint between the ears to the snout.
# The coordinate ``phi`` is the orientation of the head vector relative to the
# positive x-axis, and ranges from :math:`-\pi` to :math:`\pi` in radians.
# (following the `atan2 <https://en.wikipedia.org/wiki/Atan2>`_ convention).
#
# In our coordinate system, ``phi`` will be
# positive if the shortest path from the positive x-axis to the vector is
# clockwise. Conversely, ``phi`` will be negative if the shortest path from
# the positive x-axis to the vector is anti-clockwise.
#
# .. image:: ../_static/Cartesian-vs-Polar.png
#   :width: 600
#   :alt: Schematic comparing Cartesian and Polar coordinates


# %%
# We can explore which ``phi`` values are most common in the dataset with
# a polar histogram. First, let's define a custom function to plot a
# histogram in polar projection.


def plot_polar_histogram(da, bin_width_deg=15):
    """Plot a polar histogram of the data in the given DataArray.

    Parameters
    ----------
    da : xarray.DataArray
        A DataArray containing angle data in radians.
    bin_width_deg : int, optional
        Width of the bins in degrees.

    """
    n_bins = int(360 / bin_width_deg)

    # initialise figure with polar projection
    fig, ax = plt.subplots(
        1, 1, figsize=(6, 6.5), subplot_kw={"projection": "polar"}
    )

    # plot histogram using xarray's built-in histogram function
    da.plot.hist(bins=np.linspace(-np.pi, np.pi, n_bins + 1), ax=ax)

    # axes settings
    ax.set_theta_direction(-1)  # theta increases in clockwise direction
    ax.set_theta_offset(0)  # set zero at the right
    ax.set_xlabel("")  # remove default x-label from xarray's plot.hist()

    # set xticks to match the phi values in degrees
    n_xtick_edges = 9
    ax.set_xticks(np.linspace(0, 2 * np.pi, n_xtick_edges)[:-1])
    xticks_in_deg = (
        list(range(0, 180 + 45, 45)) + list(range(0, -180, -45))[-1:0:-1]
    )
    ax.set_xticklabels([str(t) + "\N{DEGREE SIGN}" for t in xticks_in_deg])

    return fig, ax


# %%
# Now we can pass the ``phi`` values from the ``head_vector_polar`` array to
# the above function.

# sphinx_gallery_thumbnail_number = 2

phi_data = head_vector_polar.sel(space_pol="phi").squeeze()

fig, ax = plot_polar_histogram(phi_data, bin_width_deg=15)
ax.set_title("Head vector polar angle (phi) histogram", pad=25)
plt.show()

# %%
# The ``phi`` circular histogram shows that the head vector appears at a
# variety of orientations in this dataset.

# %%
# ``movement`` also provides a ``pol2cart`` utility to transform
# data in polar coordinates back to cartesian.
# Note that the resulting ``head_vector_cart`` array has a ``space`` dimension
# with two coordinates: ``x`` and ``y``.

head_vector_cart = pol2cart(head_vector_polar)
print(head_vector_cart)

# %%
# Compute head direction as the "forward" vector
# ---------------------------------------------
# Another way to compute the head direction is to use
# :func:`compute_forward_vector()<movement.kinematics.compute_forward_vector>`
# function, which takes a different approach to the one we used above.
#
# This function expects a pair of keypoints that are bilaterally symmetric
# and computes the vector that's perpendicular to the line connecting them.
# In our case, we can use the left and right ears as the bilaterally symmetric
# keypoints.

forward_vector = compute_forward_vector(
    position,
    left_keypoint="left_ear",
    right_keypoint="right_ear",
    camera_view="top_down",
)
print(forward_vector)

# %%
# There are several reasons why this method might be preferred over the
# previous one. For example, the ears may be more reliably tracked than the
# snout (because they are occluded more rarely), making the resulting vector
# more robust.


# %%
# Visualise the head vector
# ---------------------------
# To visually check our computation of the head vector, it is easier to select
# a subset of the data. We can focus on the trajectory of the head
# within a small time window.

# time window
time_window = range(1650, 1661)  # frames

fig, ax = plt.subplots()

# plot the computed head vector originating from the midpoint between the ears
ax.quiver(
    midpoint_ears.sel(space="x", time=time_window),
    midpoint_ears.sel(space="y", time=time_window),
    head_vector.sel(space="x", time=time_window),
    head_vector.sel(space="y", time=time_window),
    angles="xy",
    scale=1,
    scale_units="xy",
    headwidth=5,
    headlength=7,
    headaxislength=7,
    color="gray",
)

# plot midpoint between the ears within the time window
plot_trajectory(
    midpoint_ears.sel(time=time_window),
    ax=ax,
    s=20,
)
# plot the snout within the time window
plot_trajectory(
    position.sel(time=time_window),
    keypoints="snout",
    ax=ax,
    s=20,
    c="r",
    marker="*",
)

ax.set_title("Zoomed in head vector")
ax.invert_yaxis()

ax.legend(
    [
        "head vector",
        "midpoint_ears",
        "snout",
    ],
    loc="best",
)


# %%
# From the plot we can confirm the head vector goes from the midpoint between
# the ears to the snout, as we defined it.

# %%
# Polar plot of the head vector within a time window
# ---------------------------------------------------
# We can also use a polar plot to represent the head vector in time. This way
# we can visualise the head vector in a coordinate system that translates with
# the mouse but is always  parallel to the pixel coordinate system.
# Again, this will be easier to visualise if we focus on a
# small time window.

# select phi values within a time window
phi = head_vector_polar.sel(
    individuals="individual_0",
    space_pol="phi",
    time=time_window,
).values

# plot tip of the head vector within that window, and color based on time
fig = plt.figure()
ax = fig.add_subplot(projection="polar")
sc = ax.scatter(
    phi,
    np.ones_like(phi),  # assign a constant value rho=1 for visualization
    c=time_window,
    cmap="viridis",
    s=50,
)

# axes settings
ax.set_theta_direction(-1)  # theta increases in clockwise direction
ax.set_theta_offset(0)  # set zero at the right
cax = fig.colorbar(
    sc,
    ax=ax,
    label=f"time ({ds.attrs['time_unit']})",
    ticks=list(time_window)[0::2],
)

# set xticks to match the phi values in degrees
n_xtick_edges = 9
ax.set_xticks(np.linspace(0, 2 * np.pi, n_xtick_edges)[:-1])
xticks_in_deg = (
    list(range(0, 180 + 45, 45)) + list(range(0, -180, -45))[-1:0:-1]
)
ax.set_xticklabels([str(t) + "\N{DEGREE SIGN}" for t in xticks_in_deg])

fig.show()

# %%
# In the polar plot above, the midpoint between the ears is at the centre of
# the plot. The tip of the head vector (the ``snout``) is represented with
# color markers at a constant ``rho`` value of 1. Markers are colored by frame.
# The polar plot shows how in this small time window of 20 frames,
# the head of the mouse turned anti-clockwise.
