"""Compute angular velocity and acceleration
============================================

Compute the angular velocity and acceleration of the head.
"""

# %%
# Imports
# -------

# For interactive plots: install ipympl with `pip install ipympl` and uncomment
# the following line in your notebook
# %matplotlib widget
import xarray as xr
from matplotlib import pyplot as plt

from movement import sample_data
from movement.filtering import (
    filter_by_confidence,
    interpolate_over_time,
    rolling_filter,
)
from movement.kinematics import (
    compute_angular_acceleration,
    compute_angular_velocity,
    compute_forward_vector,
)

# %%
# Let's set some xarray display options to make the outputs less verbose.

xr.set_options(
    display_expand_attrs=False,
    display_expand_coords=False,
)

# %%
# Before you start
# ----------------
# This example builds on the
# :ref:`sphx_glr_examples_compute_head_direction.py` example, which explains
# how ``movement`` represents orientations as vectors and as angles.
# We recommend reading that one first.
#
# Here we focus on how fast an orientation changes over time, i.e. its
# *angular velocity* and *angular acceleration*. While we use the head
# direction of a freely moving mouse as an example, the same functions work
# for any 2D orientation represented as vectors.

# %%
# Preparing the data
# ------------------
# We use the same single-mouse dataset as in the head direction example,
# i.e. a top-down video of a mouse exploring an elevated plus maze.

ds = sample_data.fetch_dataset("DLC_single-mouse_EPM.predictions.h5")
print(f"fps: {ds.fps}")
ds

# %%
# Next we remove low-confidence predictions, apply a rolling median filter
# to suppress tracking outliers, and interpolate to fill short gaps in the
# data.
# Each operation here overwrites the ``position`` variable in the dataset.
# We select that variable and drop the "individual" dimension for convenience,
# as we are only working with one mouse here.
#
# .. seealso::
#   For more details on these preprocessing steps, see the
#   :ref:`sphx_glr_examples_filter_and_interpolate.py` and
#   :ref:`sphx_glr_examples_smooth.py` examples.

# preprocessing
ds["position"] = filter_by_confidence(
    ds.position,
    ds.confidence,
    threshold=0.9,
)
ds["position"] = rolling_filter(
    ds.position,
    window=5,
    min_periods=2,
    statistic="median",
)
ds["position"] = interpolate_over_time(ds.position, max_gap=2)

# remove the "individual" dimension
position = ds.position.drop_vars("individual").squeeze("individual")

position

# %%
# We compute the head vector from the left and right ear keypoints,
# as in :ref:`sphx_glr_examples_compute_head_direction.py`.

head_vector = compute_forward_vector(
    position,
    left_keypoint="left_ear",
    right_keypoint="right_ear",
    camera_view="top_down",
)
head_vector

# %%
# The ``head_vector`` array has the same number of time points as the
# ``position`` array it was derived from, and two spatial dimensions (x and y).

# %%
# Compute angular velocity
# ------------------------
# We compute the angular head velocity (AHV) by passing the ``head_vector``
# array to :func:`~movement.kinematics.compute_angular_velocity()`.

ahv = compute_angular_velocity(head_vector)
ahv

# %%
# We get a 1D time-varying array in radians per time unit (here seconds);
# the result can also be returned in degrees per time unit, as we do below.
# We can define some metadata attributes to keep track of the units and
# a descriptive variable name; these are useful for plotting.

ahv_deg = compute_angular_velocity(head_vector, in_degrees=True)
ahv_deg.attrs["long_name"] = "Angular head velocity"
ahv_deg.attrs["units"] = "deg/s"
ahv_deg

# %%
# Let's look at a histogram of angular head velocities.
# We expect most values to be small and distributed symmetrically around zero,
# with a few extreme values in the tails. To make the tails more visible,
# we also look at the histogram with a logarithmic y-axis.

N_BINS = 100

fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharex=True)
ahv_deg.plot.hist(bins=N_BINS, ax=axes[0], color="grey")
ahv_deg.plot.hist(bins=N_BINS, ax=axes[1], yscale="log", color="grey")
axes[0].set_title("Linear y-axis")
axes[1].set_title("Logarithmic y-axis")
plt.tight_layout()
plt.show()

# %%
# .. admonition:: Sign convention
#   :class: note
#
#   Positive angular velocities correspond to rotations from the positive
#   x-axis towards the positive y-axis. In image coordinates (y pointing
#   down) with a top-down camera, that is a *clockwise* rotation on screen.
#
# We see a few values beyond :math:`\pm` 1000 deg/s, and some even
# beyond :math:`\pm` 2000 deg/s. Mice can turn their heads very quickly,
# but some of these extreme values are likely due to noise.
#
# Next we'll look at ways to smooth the AHV estimate.

# %%
# Smooth angular velocity
# -----------------------
# The angular velocity signal is noisy, because differentiation amplifies
# small frame-to-frame jitter. The preprocessing we have already applied to the
# ``position`` data helps with that, but may not be sufficient. We can
# apply some additional smoothing in one of two places:
#
# **A. Pre-smoothing.** Smooth the vector with a rolling mean before
# differentiating. As an aside, the angle of that smoothed head vector at each
# time point equals the circular mean of head angles within a window centred
# on that time point. Avoid taking the rolling mean of angles directly
# (as opposed to 2D vectors); that would give incorrect results near
# :math:`\pm\pi`.
#
# **B. Windowed estimate.** Passing ``window`` (an odd number of
# frames) to :func:`~movement.kinematics.compute_angular_velocity()` fits a
# straight line to the head angle over that many frames, centred on each time
# point, and returns its slope. This **least-squares estimate** requires
# data uniformly sampled in time. It uses every frame in the window, returning
# NaN wherever the window is incomplete; this occurs in the first and last
# ``window // 2`` frames, and around missing values.
#
# Let's compare the two approaches, and their combination. Feel free to play
# with ``WINDOW_VEC``, ``WINDOW_LSQ`` and ``TIME_SLICE`` to see how the
# results change.

WINDOW_VEC = 3  # in frames, used for pre-smoothing the vector
WINDOW_LSQ = 5  # in frames, used for the least-squares slope estimation
TIME_SLICE = slice(80, 90)  # in seconds, controls what's plotted

head_vector_smooth = rolling_filter(head_vector, WINDOW_VEC, statistic="mean")
ahv_presmoothed = compute_angular_velocity(head_vector_smooth, in_degrees=True)
ahv_lsq = compute_angular_velocity(
    head_vector, window=WINDOW_LSQ, in_degrees=True
)
ahv_presmoothed_lsq = compute_angular_velocity(
    head_vector_smooth, window=WINDOW_LSQ, in_degrees=True
)

label_to_da = {
    "raw": ahv_deg,
    "A: smoothed vector": ahv_presmoothed,
    "B: least-squares": ahv_lsq,
    "A + B": ahv_presmoothed_lsq,
}
colors = ["grey", "tab:blue", "tab:red", "tab:purple"]

fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True, sharey=True)
for i, ax in enumerate(axes):
    title, da = list(label_to_da.items())[i + 1]
    color = colors[i + 1]
    ahv_deg.sel(time=TIME_SLICE).plot(ax=ax, color=colors[0], alpha=0.5)
    da.sel(time=TIME_SLICE).plot(ax=ax, color=color, alpha=0.8)
    ax.set_xlim(TIME_SLICE.start, TIME_SLICE.stop)
    ax.set_title(title, color=color)
    ax.set_xlabel("")
    ax.set_ylabel(ahv_deg.attrs["units"])

axes[-1].set_xlabel("Time (s)")
fig.suptitle(f"{ahv_deg.attrs['long_name']} (raw vs smoothed)")
fig.tight_layout()
plt.show()

# %%
# All three approaches achieve similar effects, dampening the extreme values.
# They differ in the degree of dampening (which also depends on window sizes)
# and in the exact shape of the smoothed signal peaks and troughs.
# Let's look at the distribution of angular head velocities after smoothing,
# and compare it to the raw distribution.

# sphinx_gallery_thumbnail_number = 3

raw_range = (ahv_deg.min().item(), ahv_deg.max().item())  # same bins for all

fig, axes = plt.subplots(1, 4, figsize=(8, 3), sharex=True, sharey=True)
for i, ax in enumerate(axes):
    title, da = list(label_to_da.items())[i]
    da.plot.hist(
        ax=ax, bins=N_BINS, range=raw_range, yscale="log", color=colors[i]
    )
    ax.set_title(title, color=colors[i])
    ax.set_xlabel(ahv_deg.attrs["units"])
fig.suptitle(f"Distribution of {ahv_deg.attrs['long_name']}")
fig.tight_layout()
plt.show()

# %%
# Some extreme values beyond :math:`\pm` 2000 deg/s have survived the
# pre-smoothing approach (A), but the least-squares approach (B) and its
# combination with pre-smoothing (A + B) have removed them.

# %%
# .. admonition:: Centred vs trailing window estimate
#   :class: note
#
#   ``movement`` uses *centred* windows, so smoothing does not shift the
#   signal in time. Some studies may need to compute the derivative over a
#   window *ending* on each time point. We can get that trailing estimate
#   by shifting the centred one forward by ``window // 2`` frames:
#
#   .. code-block:: python
#
#       ahv_lsq_trailing = ahv_lsq.shift(time=WINDOW_LSQ // 2)

# %%
# Compute angular acceleration
# ----------------------------
# :func:`~movement.kinematics.compute_angular_acceleration()` takes the same
# arguments as :func:`~movement.kinematics.compute_angular_velocity()`.
# If ``window`` is supplied, it fits a quadratic to the head angle over the
# window and returns its second derivative. For this, ``window`` must be
# odd and greater than 2. Feel free to play with ``WINDOW_LSQ_ACC`` to see
# how the results change.

WINDOW_LSQ_ACC = 7  # in frames, used for the least-squares quadratic fit

aha_deg = compute_angular_acceleration(head_vector, in_degrees=True)
aha_deg.attrs["long_name"] = "Angular head acceleration"
aha_deg.attrs["units"] = "deg/s²"

aha_lsq = compute_angular_acceleration(
    head_vector, window=WINDOW_LSQ_ACC, in_degrees=True
)

label_to_da_acc = {"raw": aha_deg, "B: least-squares": aha_lsq}
colors_acc = ["grey", "tab:red"]

fig, ax = plt.subplots(figsize=(8, 3))
for i, (label, da) in enumerate(label_to_da_acc.items()):
    color = colors_acc[i]
    da.sel(time=TIME_SLICE).plot(ax=ax, color=color, alpha=0.8, label=label)

ax.set_xlim(TIME_SLICE.start, TIME_SLICE.stop)
ax.set_title(aha_deg.attrs["long_name"])
ax.set_xlabel("Time (s)")
ax.set_ylabel(aha_deg.attrs["units"])
ax.legend()
fig.tight_layout()
plt.show()

# %%
# Differentiating twice amplifies frame-to-frame jitter even more than
# differentiating once, so the raw angular acceleration is dominated by
# noise. Smoothing, via a least-squares approach and/or by pre-smoothing the
# vector, is therefore even more important here than for angular velocity.
