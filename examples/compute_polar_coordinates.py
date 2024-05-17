"""Express 2D vectors in polar coordinates
============================================

Compute a vector representing head direction and express it in polar
coordinates.
"""

# %%
# Imports
# -------

# For interactive plots: install ipympl with `pip install ipympl` and uncomment
# the following line in your notebook
# %matplotlib widget
import numpy as np
from matplotlib import pyplot as plt

from movement.io import load_poses
from movement.utils.vector import cart2pol  # norm missing?

# %%
# Load sample dataset
# ------------------------
# In this tutorial, we will use a sample dataset with a single individual and
# six keypoints.


# ds = sample_data.fetch_dataset(
#     "SLEAP_single-mouse_EPM.predictions.slp",
# )
# ------------------ replace after train
ds = load_poses.from_file(
    "/Users/sofia/.movement/data/poses/SLEAP_single-mouse_EPM.predictions.slp",
    source_software="SLEAP",
)
# -----------------------------

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
# Compute head vector
# ---------------------
# To demonstrate how polar coordinates can be useful in behaviour analyses, we
# will compute the head vector of the mouse.
# We define the head vector as the vector from the midpoint between the ears
# to the snout.

# compute the midpoint between the ears
midpoint_ears = 0.5 * (
    position.sel(keypoints="left_ear") + position.sel(keypoints="right_ear")
)

# compute the head vector
head_vector = position.sel(keypoints="snout") - midpoint_ears
head_vector.drop_vars("keypoints")  # drop the keypoints dimension

# %%
# Visualise the head trajectory
# ---------------------------------
# We can plot the data to check that our computation of the head vector is
# correct.
#
# We can start by plotting the trajectory of the midpoint between the ears.

fig, ax = plt.subplots(1, 1)
mouse_name = ds.individuals.values[0]  # 'track_0'

sc = ax.scatter(
    midpoint_ears.sel(individuals=mouse_name, space="x"),
    midpoint_ears.sel(individuals=mouse_name, space="y"),
    s=15,
    c=midpoint_ears.time,
    cmap="viridis",
    marker="o",
)

ax.axis("equal")
ax.set_xlabel("x (pixels)")
ax.set_ylabel("y (pixels)")
ax.invert_yaxis()
ax.set_title(f"Head trajectory ({mouse_name})")
fig.colorbar(sc, ax=ax, label=f"time ({ds.attrs['time_unit']})")

# %%
# In this dataset the mouse is moving on an
# [Elevated Plus Maze](https://en.wikipedia.org/wiki/Elevated_plus_maze) and
# we can see that in the head trajectory plot.

# %%
# Visualise head vector
# ---------------------------
# To check our computation of the head vector, it is easier to plot only a
# subset of the data. We can focus on the trajectory of the head when the
# mouse is within a small rectangular area, and within a certain time window.

# area of interest
xmin, ymin = 600, 665  # pixels
x_delta, y_delta = 125, 100  # pixels

# time window
time_window = range(1650, 1670)  # frames

# %%
# For that subset of the data, we now plot the head vector.

fig, ax = plt.subplots(1, 1)
mouse_name = ds.individuals.values[0]

# plot midpoint between the ears and color based on time
sc = ax.scatter(
    midpoint_ears.sel(individuals=mouse_name, space="x", time=time_window),
    midpoint_ears.sel(individuals=mouse_name, space="y", time=time_window),
    s=50,
    c=midpoint_ears.time[time_window],
    cmap="viridis",
    marker="*",
)

# plot snout and color based on time
sc = ax.scatter(
    position.sel(
        individuals=mouse_name, space="x", time=time_window, keypoints="snout"
    ),
    position.sel(
        individuals=mouse_name, space="y", time=time_window, keypoints="snout"
    ),
    s=50,
    c=position.time[time_window],
    cmap="viridis",
    marker="o",
)

# plot the computed head vector
ax.quiver(
    midpoint_ears.sel(individuals=mouse_name, space="x", time=time_window),
    midpoint_ears.sel(individuals=mouse_name, space="y", time=time_window),
    head_vector.sel(individuals=mouse_name, space="x", time=time_window),
    head_vector.sel(individuals=mouse_name, space="y", time=time_window),
    angles="xy",
    scale=1,
    scale_units="xy",
    headwidth=7,
    headlength=9,
    headaxislength=9,
    color="gray",
)

ax.axis("equal")
ax.set_xlim(xmin, xmin + x_delta)
ax.set_ylim(ymin, ymin + y_delta)
ax.set_xlabel("x (pixels)")
ax.set_ylabel("y (pixels)")
ax.set_title(f"Zoomed in to check ({mouse_name})")
ax.invert_yaxis()
fig.colorbar(sc, ax=ax, label=f"time ({ds.attrs['time_unit']})")

ax.legend(
    [
        "midpoint_ears",
        "snout",
        "head_vector",
    ],
    loc="best",
)

fig.show()

# %%
# From the plot we can confirm the head vector goes from the midpoint between
# the ears to the snout, as we defined it.


# %%
# Express head vector in polar coordinates
# -------------------------------------------------------------
# Link to polar:
# a convenient way to work with vectors orientations in 2D is
# with polar coordinates....
head_vector_pol = cart2pol(head_vector)

# Explain rho and theta, and in our case
# rho: norm; here distance between midpoint btw ears and snout per frame
# theta: angle of the vector wrt x positive, increasing from xpositive to y
# positive (cloclwise) and ranging from -pi to pi


# %%
# Rho histogram for the full data
# -------------------------------------------------------------
# rho should be approx constant -- check our assumption
fig, ax = plt.subplots(1, 1)
# ax.plot(head_vector_pol[:,0,0])
# ax.plot(head_vector_pol[:,0,1])

ax.hist(head_vector_pol[:, 0, 0], bins=50)
np.nanmedian(head_vector_pol[:, 0, 0])
np.nanmean(head_vector_pol[:, 0, 0])

# %%
# Theta histogram for the full dataset
# -------------------------------------

fig = plt.figure()
ax = fig.add_subplot(projection="polar")


# Create a histogram
# Number of bins

bin_width_deg = 5  # deg
num_bins = int(360 / bin_width_deg)

# Histogram of theta
slc_not_nan = (
    ~np.isnan(
        head_vector_pol.sel(individuals=mouse_name, space_pol="phi").values
    ),
)
counts, bins = np.histogram(
    head_vector_pol.sel(individuals=mouse_name, space_pol="phi").values[
        slc_not_nan
    ],
    bins=np.linspace(-np.pi, np.pi, num_bins + 1),
    # np.linspace(-np.pi-np.deg2rad(bin_width/2),
    # np.pi+np.deg2rad(bin_width/2), num_bins+1),
    # weights=head_vector_pol.sel(individuals=mouse_name,
    # space_pol="rho").values[slc_not_nan]
)

# Plot the histogram as a bar plot
bin_width_rad = np.deg2rad(bin_width_deg)  # np.diff(bins)
bars = ax.bar(bins[:-1], counts, width=bin_width_rad, align="edge")

# Optionally, customize the plot (e.g., title, labels)
ax.set_title("Theta histogram")
ax.set_theta_direction(-1)  # set direction clockwise
ax.set_theta_offset(0)  # set zero at the left

# %%
# The theta circular histogram shows that the head vector appears at a variety
# of orientations in this clip. The histogram counts peak at 90deg and 270deg,
# which is consistent with the animal spending more time in the vertical
# arm of the maze, walking along it with its head looking forward.

# change tick labels to match the theta values?
# Could also do the body vector and compare the two arms... maybe more evident

# %%
# Polar plot of head unit vector in selected time window
# -------------------------------------------------------------
# python polar plot: https://www.geeksforgeeks.org/plotting-polar-curves-in-python/
# shows how the vector changes in time, in a coordinate system that is parallel
# to the world cs and moves with the head
# overlay on quiver plot?

theta = head_vector_pol.sel(
    individuals=mouse_name,
    space_pol="phi",
    time=time_window,
).values

fig = plt.figure()
ax = fig.add_subplot(projection="polar")
sc = ax.scatter(
    theta,
    np.ones_like(theta),
    c=time_window,
    cmap="viridis",
    s=50,
)
ax.set_theta_direction(-1)  # set direction counterclockwise
ax.set_theta_offset(0)  # set zero at the left
cax = fig.colorbar(
    sc,
    ax=ax,
    label=f"time ({ds.attrs['time_unit']})",
    ticks=list(time_window)[0::2],
)


# Note that
# - angles are measured from x-positive in counterclockwise direction
# - values are in radians and wrapped from -pi to pi (but expressed in the plot
#  from 0 to 360 deg)


# %%
# mention also the inverse? (pol2cart)
# suggest to name theta by default? (matplotlib uses theta)
