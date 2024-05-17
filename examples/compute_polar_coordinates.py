"""Compute polar coordinates for 2D data.
====================================

Compute ....
"""

# %%
# Imports
# -------

import numpy as np

# For interactive plots: install ipympl with `pip install ipympl` and uncomment
# the following line in your notebook
# %matplotlib ipympl
# widget
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from movement.io import load_poses
from movement.utils.vector import cart2pol  # norm missing?

# %%
# Load sample dataset
# ------------------------
# First, we load an example dataset. In this case, we will use
# ``SLEAP_single-mouse_EPM.predictions``.
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
# %%
# The metadata shows that the dataset contains data for one individual
# ('track_0')
# and 6 keypoints
print(f"Individuals: {ds.individuals.values}")
print(f"Keypoints: {ds.keypoints.values}")


# %%
# The loaded dataset ``ds`` contains two data variables:
# ``position`` and ``confidence``. In this tutorial, we will use the
# ``position`` data array:
position = ds.position


# %%
# Compute head vector
# ---------------------
# To demonstrate how polar coordinates can be useful in behaviour analyses, we
# will compute the head vector of the mouse.
# We define the head vector as the vector from the midpoint between the ears
# to the snout.

midpoint_ears = 0.5 * (
    position.sel(keypoints="left_ear") + position.sel(keypoints="right_ear")
)  # returns a view?
head_vector = (
    position.sel(keypoints="snout") - midpoint_ears
)  # why does it have snout?

head_vector.drop_vars("keypoints")

# %% Visualise the head trajectory
# ---------------------
# We can plot the data to check our computation
fig, ax = plt.subplots(1, 1)
mouse_name = ds.individuals.values[0]

# plot midpoint ears full trajectory: the mouse is moving on a elevated
# platform maze
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
ax.set_title(f"Midpoint between the ears ({mouse_name})")
fig.colorbar(sc, ax=ax, label=f"time ({ds.attrs['time_unit']})")

# %%
# Area of focus in space and time
# ---------------------
# Create a Rectangle patch
xmin, ymin = 600, 665
x_delta, y_delta = 125, 100
rect = Rectangle(
    (xmin, ymin),
    x_delta,
    y_delta,
    linewidth=1,
    edgecolor="r",
    facecolor="none",
)

# Add the patch to the Axes
ax.add_patch(rect)

# time window in red
time_window = range(1650, 1670)  # frames
# bool_x_in_box = (midpoint_ears[:,0,0] >= xmin) & (midpoint_ears[:,0,0]
# <= xmin+x_delta + 1)
# bool_y_in_box = (midpoint_ears[:,0,1] >= ymin) &( midpoint_ears[:,0,1]
# <= ymin+y_delta + 1)
sc = ax.scatter(
    midpoint_ears.sel(
        individuals=mouse_name, time=time_window, space="x"
    ),  # .where(bool_x_in_box),
    midpoint_ears.sel(
        individuals=mouse_name, time=time_window, space="y"
    ),  # .where(bool_y_in_box),
    s=15,
    c="r",
    marker="o",
)

# %%
# Plot zoomed in head vector
# ---------------------
# plot midpoint ears: small time window and zoom in
fig, ax = plt.subplots(1, 1)
mouse_name = ds.individuals.values[0]


# midpoint ears
sc = ax.scatter(
    midpoint_ears.sel(individuals=mouse_name, space="x", time=time_window),
    midpoint_ears.sel(individuals=mouse_name, space="y", time=time_window),
    s=50,
    c=midpoint_ears.time[time_window],
    cmap="viridis",
    marker="*",
)

# snout
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

# head vector
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

# %% In relation to other keypoints

# plot 'snout', 'left_ear', 'right_ear' keypoints for the middle frame
middle_frame = int(np.median(time_window))

ax.quiver(
    midpoint_ears.sel(individuals=mouse_name, space="x", time=middle_frame),
    midpoint_ears.sel(individuals=mouse_name, space="y", time=middle_frame),
    head_vector.sel(individuals=mouse_name, space="x", time=middle_frame),
    head_vector.sel(individuals=mouse_name, space="y", time=middle_frame),
    angles="xy",
    scale=1,
    scale_units="xy",
    headwidth=7,
    headlength=9,
    headaxislength=9,
    color="r",
)

# plot 'snout', 'left_ear', 'right_ear' keypoints for the middle frame
sc = ax.scatter(
    position.sel(
        individuals=mouse_name, space="x", time=middle_frame
    ),  # , keypoints=['snout', 'left_ear', 'right_ear']),
    position.sel(
        individuals=mouse_name, space="y", time=middle_frame
    ),  # , keypoints=['snout', 'left_ear', 'right_ear']),
    s=205,
    c="r",
    marker=".",
)

for kpt in position.keypoints.values:
    ax.text(
        1.005
        * position.sel(
            individuals=mouse_name, space="x", time=middle_frame, keypoints=kpt
        ),
        1.005
        * position.sel(
            individuals=mouse_name, space="y", time=middle_frame, keypoints=kpt
        ),
        kpt,
        c="r",
    )

# ax.legend(
#     [
#         'midpoint_ears',
#         'snout',
#         'head_vector', f'head_vector (frame={middle_frame})',
# f'keypoints (frame={middle_frame})'
#     ],
#     loc='best'
# )

# %%
# Quiver plot of unit head vector with vectors always at 0,0
# -------------------------------------------
# like [compass](https://uk.mathworks.com/help/matlab/ref/compass.html) plot
# in matlab


# Link to polar:
# a convenient way to work with vectors orientations in 2D is
# with polar coordinates....
# theta is the angle with the x-axis ("bottom side" of the image OJO)
# How can we visualise how the head vector changes in time?

# %%
# Express head vector in polar coordinates
# -------------------------------------------------------------

head_vector_pol = cart2pol(head_vector)


# %%
# Rho histogram
# -------------------------------------------------------------
# rho should be approx constant -- check our assumption
fig, ax = plt.subplots(1, 1)
# ax.plot(head_vector_pol[:,0,0])
# ax.plot(head_vector_pol[:,0,1])

ax.hist(head_vector_pol[:, 0, 0], bins=50)
np.nanmedian(head_vector_pol[:, 0, 0])
np.nanmean(head_vector_pol[:, 0, 0])

# %%
# Polar plot
# -------------------------------------------------------------
# python polar plot: https://www.geeksforgeeks.org/plotting-polar-curves-in-python/
# shows how the vector changes in time, in a coordinate system that is parallel
# to the world cs and moves with the head
# overlay on quiver plot?

fig = plt.figure()
ax = fig.add_subplot(projection="polar")
ax.scatter(
    head_vector_pol.sel(
        individuals=mouse_name, space_pol="phi"
    ),  # suggest to name theta by default? (matplotlib uses theta)
    head_vector_pol.sel(individuals=mouse_name, space_pol="rho"),
    c=head_vector_pol.time,
    cmap="viridis",
    s=5,
    alpha=0.5,
)


# %%

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
bin_width_rad = np.diff(bins)
bars = ax.bar(bins[:-1], counts, width=bin_width_rad, align="edge")

# Optionally, customize the plot (e.g., title, labels)
ax.set_title("Theta histogram")
ax.set_theta_direction(1)  # set direction counterclockwise
ax.set_theta_offset(0)  # set zero at the left


# %%
# mention also the inverse? (pol2cart)
