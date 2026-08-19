"""Zebra escape trajectories
============================

Quantify the collective behaviour of a herd of zebras in escape response.
"""

# %%
# .. admonition:: Acknowledgements
#   :class: acknowledgements
#
#   The sample video and original SLEAP trajectories were kindly shared by
#   `Dr. Isla Duporge <https://eeb.princeton.edu/people/isla-duporge>`_ from
#   the Rubenstein Lab at Princeton University. The trajectories in the world
#   coordinate system were computed by Sofía Miñano, Niko Sirmpilatze and
#   Igor Tatarnikov, and the analysis below follows the
#   `zebras-stitching <https://github.com/neuroinformatics-unit/zebras-stitching>`_
#   repository.

# %%
# Overview
# --------
# In this example we will be using ``movement`` to quantify the collective
# behaviour of a herd of zebras in escape response. The data we will use is
# part of a larger dataset collected in Mpala (Kenya), in which researchers
# simulated predation events to study the group response of the animals. We
# will demonstrate how we can compute useful metrics for this analysis using
# ``movement``.

# %%
# Imports
# -------

# For interactive plots: install ipympl with `pip install ipympl` and uncomment
# the following line in your notebook
# %matplotlib widget
import matplotlib.pyplot as plt
import numpy as np

from movement import sample_data
from movement.kinematics import compute_speed
from movement.transforms import scale
from movement.utils.reports import report_nan_values
from movement.utils.vector import compute_norm, convert_to_unit

# %%
# Load the dataset
# ----------------
# The 3.5-min dataset presented here consists of 44 trajectories of zebras
# (*Equus quagga*) expressed in a coordinate system fixed to the ground. Each
# individual has two keypoints (head and tail).
#
# The data was collected with a camera drone that followed the herd and
# recorded video data. Note that since both camera and animals are in motion,
# applying pose estimation directly to the video data would confound both
# sources of movement. We need the trajectories of the animals relative to the
# ground, rather than relative to the drone. We computed the ground
# trajectories as follows: first, a trained `SLEAP <https://sleap.ai/>`_ model
# was run on a video clip recorded from the drone. This way we obtained the
# trajectories of the zebras in a coordinate system linked to the drone. Then,
# we computed the position and orientation of the camera drone at each
# timestep by applying `structure-from-motion
# <https://en.wikipedia.org/wiki/Structure_from_motion>`_ (with `OpenSfM
# <https://github.com/mapillary/OpenSfM>`_ and `OpenDroneMap
# <https://github.com/OpenDroneMap/OpenDroneMap>`_). This allowed us to
# express the trajectories in a coordinate system fixed to the ground, and to
# disentangle the motion of the zebras from the movement of the camera drone.
# After this coordinate transformation, the data was cleaned by removing
# low-confidence keypoints and implausible data points.

ds = sample_data.fetch_dataset("SLEAP_OSFM_zebras_drone.h5")
print(ds)

# %%
# We can see the poses dataset ``ds`` is made up of two data arrays,
# ``position`` and ``confidence``. In this example, we will use the
# ``position`` data array only, which spans four dimensions: ``time``,
# ``space``, ``keypoint`` and ``individual``. We can verify there are 44
# individuals in this dataset (``track_0`` to ``track_43``) and two keypoints
# per individual, labelled ``H`` (head) and ``T`` (tailbase). The data was
# collected at 29.97 frames per second, and the dataloader used this
# information to automatically express the ``time`` dimension in seconds.
#
# .. note::
#   The ``position`` data in ``ds`` is expressed in arbitrary units. This is
#   because no GPS data was available for georeferencing or defining ground
#   control points in the structure-from-motion (SfM) analysis. As a result,
#   the scale factor remains a free parameter in the reconstruction of the
#   world coordinates.
#
#   Note however that this will not be a problem for our analysis, since the
#   relative positions between the individuals are still correct. Moreover, we
#   will use the median zebra body length to scale the data to more
#   informative units. For more details on the coordinate systems involved in
#   SfM analysis see the `OpenSfM documentation
#   <https://opensfm.org/docs/geometry.html#world-coordinates>`_.

# %%
# Compute the body length per individual
# --------------------------------------
# We define the body vector for each individual as the vector going from the
# ``T`` keypoint (tail) to the ``H`` keypoint (head).
body_vector = ds.position.sel(keypoint="H") - ds.position.sel(keypoint="T")

print(body_vector)

# %%
# We can compute the body length of each individual by computing the norm of
# the body vector, using :func:`movement.utils.vector.compute_norm`.
body_length = compute_norm(body_vector)

# %%
# It would be useful to check if there are missing values in the body length
# array. We can quickly inspect this using ``movement``'s
# :func:`movement.utils.reports.report_nan_values` function.
print(report_nan_values(body_length))

# %%
# The output shows that the number of missing values per individual varies
# between 0.27% and 19.92%. This is not necessarily a problem for our
# analysis, but it is something to keep in mind when interpreting the results.
# These missing points are likely due to imperfect tracking of one or both of
# the keypoints required to compute the body vector.

# %%
# Let's compute some basic statistics to get a sense of the distribution of
# the body length values.
body_length_std = body_length.std()
body_length_mean = body_length.mean()
body_length_median = body_length.median()

# a.u.: arbitrary units
print(f"Body length mean: {body_length_mean:.2f} a.u.")
print(f"Body length median: {body_length_median:.2f} a.u.")
print(f"Body length std: {body_length_std:.2f} a.u.")

# %%
# We can also plot the distribution of body lengths.
fig, ax = plt.subplots()

# plot histogram of body length values
counts, bins, _ = body_length.plot.hist(bins=100)

# add reference lines for mean and mean +- 2 stds
ax.vlines(
    body_length_mean,
    ymin=0,
    ymax=np.max(counts),
    color="red",
    linestyle="-",
    label="mean body length",
)
lower_bound = body_length_mean - 2 * body_length_std
upper_bound = body_length_mean + 2 * body_length_std
for bound in [lower_bound, upper_bound]:
    ax.vlines(
        bound,
        ymin=0,
        ymax=np.max(counts),
        color="red",
        linestyle="--",
        label="mean +- 2 std",
    )
ax.set_ylim(0, np.max(counts))
ax.set_xlabel("body length (a.u.)")
ax.set_ylabel("counts")
ax.legend()
plt.show()

# %%
# We can see there is some variability in the body lengths per individual.
# Part of it may reflect the diversity across individuals, but from visual
# inspection of the video we expect the majority of it to be due to imperfect
# tracking of the keypoints. To remove some of these outliers, we continue the
# analysis considering only the samples in which an individual's body length
# is within 2 standard deviations of the mean.
within_2_stds = np.logical_and(
    body_length <= body_length_mean + 2 * body_length_std,
    body_length >= body_length_mean - 2 * body_length_std,
)

# %%
# We apply the mask to the ``position`` data array itself, so that every
# metric we compute from here on is based on the same cleaned data.

# `within_2_stds` has (time, individual) dimensions, so it broadcasts
# across the space and keypoint dimensions of the position array
position_filtered = ds.position.where(within_2_stds)

# %%
# The filtered body vectors then follow from the filtered positions.
body_vector_filtered = position_filtered.sel(
    keypoint="H"
) - position_filtered.sel(keypoint="T")

# %%
# Compute polarization
# --------------------
# We would now like to inspect the orientation of each individual in relation
# to the group while the simulated escape events take place.
#
# For this, we first compute each animal's **unit body vector**. These are a
# scaled version of the body vectors we just computed, normalised to have unit
# length. ``movement`` provides a convenience function to do this,
# :func:`movement.utils.vector.convert_to_unit`.
body_vector_filtered_unit = convert_to_unit(body_vector_filtered)

# %%
# We can quickly check if their norms are now equal to 1.
print(compute_norm(body_vector_filtered_unit))

# %%
# We now define the **herd vector** as the mean of the unit body vectors
# across all individuals detected per frame. The mean vector of a set of
# :math:`n` vectors is the sum of all the vectors (i.e., the resultant vector)
# scaled by :math:`1/n`.
herd_vector = body_vector_filtered_unit.mean("individual")
print(herd_vector)

# %%
# The resulting array has ``(time, space)`` dimensions, which means that we
# have a single herd vector defined at each timestep.
#
# The norm of the herd vector will be bounded between 0 and 1, because it is
# the mean of a set of unit vectors. This is convenient because it already
# gives us an intuition of how aligned the whole herd is. When the herd vector
# norm is close to 1, it means that the majority of the unit body vectors are
# aligned. When its norm is close to 0, it means that the unit body vectors
# are dispersed. The norm of the herd vector is sometimes called
# **polarization**.
polarization = compute_norm(herd_vector)

# %%
# We can plot the evolution of the polarization over time to get a sense of
# how the herd's alignment changes.
fig, ax = plt.subplots()
ax.plot(herd_vector.time, polarization)
ax.set_ylabel("polarization")
ax.set_xlabel("time (s)")
ax.grid()
plt.show()

# %%
# The plot suggests that the herd alternates between periods of higher and
# lower polarization.

# %%
# To confirm that these fluctuations correspond to the herd's actual
# orientations, we can additionally create an animation that plots each
# individual's unit body vector and the herd vector for every frame, and
# present this alongside the polarization-over-time plot and the drone
# footage. The resulting video is shown below. The top-left panel shows the
# polarization trace, while the top-right panel shows the per-frame
# visualisation of the herd's orientation: the black arrows are the unit body
# vectors, the red arrow is the herd vector (whose norm is the polarization
# value), and the purple line under the herd vector is the equivalent unit
# vector.
#
# .. raw:: html
#
#   <video
#     src="https://github.com/neuroinformatics-unit/zebras-stitching/releases/download/zebras-clip/zebras_20250912_174038_w1440_crf23.mp4"
#     controls loop muted playsinline width="100%"></video>

# %%
# Compute average speed of the herd
# ---------------------------------
# We can also inspect how the speed of the herd changes over the course of the
# simulated escape events.
#
# First, let's scale the filtered position data to express it in units of body
# lengths (BL). This will make the results more interpretable. We can use
# ``movement``'s :func:`movement.transforms.scale` function to do this.
position_scaled = scale(
    position_filtered,
    factor=1 / body_length_median.item(),
    space_unit="body_length",
)

# %%
# Note that the scaling factor is the median body length computed over the
# *unfiltered* data. The median is quite robust to outliers, so we leave it as
# is since recomputing it after filtering would make little difference.

# %%
# For simplicity, we would also like to reduce the position of each individual
# to a single point. A good candidate for this is the centroid, which is the
# mean of all the keypoints per individual. In our case, the centroid will be
# the midpoint between the head and tail keypoints.
centroid = position_scaled.mean("keypoint")

# %%
# We can now compute the speed of each individual's centroid with
# :func:`movement.kinematics.compute_speed`, and average it across all
# individuals to obtain the speed of the herd.
centroid_speed = compute_speed(centroid)
herd_speed = centroid_speed.mean("individual")

fig, ax = plt.subplots()
ax.plot(herd_speed.time, herd_speed)
ax.set_ylabel("herd speed (BL/s)")
ax.set_xlabel("time (s)")
ax.grid()
plt.show()

# %%
# We can see that there are four periods in the dataset in which the speed of
# the herd surpasses 2 BL/s for about 10 to 20 seconds.

# %%
# We can now plot the speed of each individual over time.
fig, ax = plt.subplots()
im = ax.matshow(
    centroid_speed,
    aspect="auto",
    cmap="viridis",
)

# convert frames to seconds in y-axis
time_ticks_step = 1498
time_ticks = np.arange(0, len(centroid_speed.time), time_ticks_step)
time_labels = [
    f"{t:.0f}" for t in centroid_speed.time.values[0:-1:time_ticks_step]
]
ax.set_yticks(time_ticks)
ax.set_yticklabels(time_labels)
ax.tick_params(
    axis="x", bottom=True, top=False, labelbottom=True, labeltop=False
)

ax.set_xlabel("individual")
ax.set_ylabel("time (s)")

# add colorbar
cbar = plt.colorbar(im)
cbar.set_label("speed (BL/s)")
ax.get_images()[0].set_clim(0, 6)  # cap values at 6 BL/s
plt.show()

# %%
# The plot suggests that the individuals change speed in a coordinated way,
# with four clear peaks matching the four simulated escape events. The white
# gaps are the samples we discarded, either because the individual was not
# detected or because its body length fell outside the range we defined above.

# %%
# Polarization vs speed
# ---------------------
# Let's now bring everything together and examine how polarization varies with
# the herd's speed.
#
# The distribution of herd speeds is skewed towards low values, so we use the
# logarithm of the speed to better resolve differences across the lower and
# middle range when colouring the points.
log10_herd_speed = np.log10(herd_speed)

# %%
# We can now plot the polarization in time, colouring the points by the
# logarithm of the speed.
fig, ax = plt.subplots()
sc = ax.scatter(
    x=polarization.time,
    y=polarization,
    c=log10_herd_speed,
    s=5,
    cmap="turbo",
    # rescale color map to 1st and 99th percentiles
    vmin=log10_herd_speed.quantile(0.01).item(),
    vmax=log10_herd_speed.quantile(0.99).item(),
)

ax.set_xlabel("time (s)")
ax.set_ylabel("polarization")

cbar = plt.colorbar(sc)
cbar.set_label("log10 herd speed (BL/s)")
plt.show()

# %%
# The plot shows that for this dataset, the periods of highest polarization
# are associated with higher speeds. This is consistent with the
# interpretation that the zebras become more aligned when escaping at speed,
# and more dispersed when they are at rest.

# %%
# .. seealso::
#   * :ref:`sphx_glr_examples_compute_kinematics.py` example.
#   * :ref:`sphx_glr_examples_scale.py` example.
#   * The `zebras-stitching
#     <https://github.com/neuroinformatics-unit/zebras-stitching>`_
#     repository, for a detailed description of how the world coordinate
#     trajectories were computed.
#   * The corresponding chapter of the `Animals in Motion
#     <https://neuroinformatics-unit.github.io/course-animals-in-motion/>`_
#     course, from which this example is adapted.

# sphinx_gallery_thumbnail_number = 4
