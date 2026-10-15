"""Compute angular velocity and acceleration
============================================

Compute the angular velocity and acceleration of the head, and smooth
them.
"""

# %%
# Imports
# -------

# For interactive plots: install ipympl with `pip install ipympl` and uncomment
# the following line in your notebook
# %matplotlib widget
from matplotlib import pyplot as plt

from movement import sample_data
from movement.filtering import rolling_filter
from movement.kinematics import (
    compute_angular_acceleration,
    compute_angular_velocity,
    compute_forward_vector,
    compute_forward_vector_angle,
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
# *angular velocity*, and on how fast that changes, i.e. its *angular
# acceleration*. We use the head direction of a mouse as an example,
# and compute its angular head velocity (AHV), a quantity commonly
# related to the activity of head-direction and AHV cells.

# %%
# Load sample dataset
# -------------------
# We use the same single-mouse dataset as in the head direction example,
# and compute the head direction (forward) vector from the two ears.

ds = sample_data.fetch_dataset("DLC_single-mouse_EPM.predictions.h5")
position = ds.position.squeeze()
fps = ds.fps
print(f"Frame rate: {fps} fps")

head_vector = compute_forward_vector(
    position, left_keypoint="left_ear", right_keypoint="right_ear"
)
head_angle = compute_forward_vector_angle(
    position, left_keypoint="left_ear", right_keypoint="right_ear"
)

# %%
# Why not simply differentiate the angle?
# ---------------------------------------
# Head direction angles live on a circle: they span :math:`(-\pi, \pi]`
# and *wrap around* when the head crosses :math:`\pm\pi`. Differentiating
# the angle naively turns every wrap-around into a huge, spurious spike.

naive = head_angle.differentiate("time")
ahv = compute_angular_velocity(head_vector)

time_window = slice(85, 90)  # seconds
fig, ax = plt.subplots(figsize=(8, 3))
naive.sel(time=time_window).plot(ax=ax, label="naive derivative of angle")
ahv.sel(time=time_window).plot(ax=ax, label="compute_angular_velocity")
ax.set_ylabel("rad/s")
ax.set_title("Naive vs wrap-safe angular velocity")
ax.legend()
fig.show()

# %%
# :func:`compute_angular_velocity()\
# <movement.kinematics.compute_angular_velocity>` handles the wrap-around
# for us. It accepts either orientation vectors (as here) or angles in
# radians, so ``compute_angular_velocity(head_angle)`` gives the same
# result.
#
# .. admonition:: Sign convention
#   :class: note
#
#   Positive angular velocities correspond to rotations from the positive
#   x-axis towards the positive y-axis. In image coordinates (y pointing
#   down) with a top-down camera, that is a *clockwise* rotation on
#   screen, i.e. a *right* turn of the animal. The sign flips for a
#   bottom-up camera.

# %%
# Smoothing angular velocity
# --------------------------
# The raw angular velocity is noisy, because differentiation amplifies
# small frame-to-frame jitter in the keypoints. There are two places where
# we can smooth.
#
# **A. Smooth the orientation before differentiating.** Smooth the
# *vector*, not the angle: a rolling mean of the vector's components is a
# proper circular mean, while a rolling mean of the angles is wrong near
# :math:`\pm\pi`. The same applies to filling gaps with
# :func:`interpolate_over_time()\
# <movement.filtering.interpolate_over_time>`.
#
# **B. Differentiate over a window.** Passing ``window`` (an odd number of
# frames) fits a straight line to the head angle over that many frames,
# centred on each time point, and returns its slope. This least-squares
# estimate uses every frame in the window. It requires uniformly sampled
# data, and returns NaN wherever the window is incomplete: in the first
# and last ``window // 2`` frames, and around missing values.
#
# **B'. Average the raw angular velocity.** Angular velocity does not wrap,
# so we can smooth it with any of our filters, e.g. a rolling mean. This
# is roughly the angle difference between the window's ends, divided by
# the window's duration.
#
# Windows are given in frames, so we convert from seconds. Note that at
# 30 fps, a 50 ms window covers only about 2 frames.

window_hd = max(1, round(0.05 * fps))  # ~50 ms
window_deriv = round(0.2 * fps) | 1  # ~200 ms, must be odd

ahv_presmoothed = compute_angular_velocity(
    rolling_filter(head_vector, window_hd, statistic="mean")
)
ahv_lsq = compute_angular_velocity(head_vector, window=window_deriv)
ahv_rolling_mean = rolling_filter(ahv, window_deriv, statistic="mean")

fig, ax = plt.subplots(figsize=(8, 3))
for da, label in [
    (ahv, "raw"),
    (ahv_presmoothed, f"A: vector smoothed ({window_hd} frames)"),
    (ahv_lsq, f"B: least-squares window ({window_deriv} frames)"),
    (ahv_rolling_mean, f"B': rolling mean ({window_deriv} frames)"),
]:
    da.sel(time=time_window).plot(ax=ax, label=label)
ax.set_ylabel("rad/s")
ax.set_title("Smoothed angular head velocity")
ax.legend(fontsize="small")
fig.show()

# %%
# Centred vs trailing windows
# ---------------------------
# ``movement`` uses *centred* windows, so smoothing does not shift the
# signal in time. Some studies instead compute the derivative over a window
# *ending* on each time point. We get that trailing estimate by shifting
# the centred one forward by ``window // 2`` frames:

ahv_trailing = ahv_lsq.shift(time=window_deriv // 2)

# %%
# Angular acceleration
# --------------------
# :func:`compute_angular_acceleration()\
# <movement.kinematics.compute_angular_acceleration>` takes the same
# arguments. Differentiating twice amplifies keypoint jitter even more,
# so the raw estimate is dominated by noise. With ``window``, it instead
# fits a quadratic to the head angle over the window and returns its
# second derivative. For this, ``window`` must be odd and greater than 2,
# and a longer window than for the velocity is usually needed.

window_accel = round(0.3 * fps) | 1  # ~300 ms, must be odd
aha_lsq = compute_angular_acceleration(head_vector, window=window_accel)

fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
ahv_lsq.sel(time=time_window).plot(ax=axes[0])
axes[0].set_ylabel("rad/s")
axes[0].set_title(f"Angular head velocity ({window_deriv}-frame window)")
axes[0].set_xlabel("")
aha_lsq.sel(time=time_window).plot(ax=axes[1])
axes[1].set_ylabel("rad/s²")
axes[1].set_title(f"Angular head acceleration ({window_accel}-frame window)")
fig.tight_layout()
fig.show()

# %%
# Acceleration peaks where the velocity changes fastest, and crosses zero
# at the velocity's peaks and troughs.

# %%
# Results in degrees
# ------------------
# Finally, ``in_degrees=True`` returns degrees per second (or per frame,
# if the ``time`` coordinate is in frames). The same option of
# :func:`compute_angular_acceleration()\
# <movement.kinematics.compute_angular_acceleration>` returns degrees per
# second squared.

ahv_deg = compute_angular_velocity(
    head_vector, window=window_deriv, in_degrees=True
)
fig, ax = plt.subplots(figsize=(5, 3))
ahv_deg.plot.hist(bins=100, ax=ax)
ax.set_xlabel("Angular head velocity (deg/s)")
ax.set_title("Distribution of angular head velocity")
fig.show()
