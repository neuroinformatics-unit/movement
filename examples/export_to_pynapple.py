"""Convert to pynapple
====================

Convert pose tracks and derived kinematic variables to ``pynapple``
objects, to analyse them alongside neural data.
"""

# %%
# Imports
# -------

# For interactive plots: install ipympl with `pip install ipympl` and uncomment
# the following line in your notebook
# %matplotlib widget
import tempfile
from pathlib import Path

import numpy as np
import pynapple as nap
import xarray as xr
from matplotlib import pyplot as plt

from movement import sample_data
from movement.filtering import filter_by_confidence, interpolate_over_time
from movement.io import to_pynapple_tsdframe
from movement.kinematics import compute_forward_vector_angle, compute_speed

# sphinx_gallery_start_ignore
xr.set_options(display_expand_attrs=False)
# sphinx_gallery_end_ignore

# %%
# Before you start
# ----------------
# `pynapple <https://pynapple.org/>`_ is a toolbox for analysing neural data.
# Its core objects are time series, such as :class:`~pynapple.TsdFrame`
# (a table of values over time, with named columns) and
# :class:`~pynapple.TsGroup` (a group of spike trains).
# Converting ``movement`` data to these objects lets you relate an animal's
# behaviour to neural activity recorded in the same session. A common
# example is computing *tuning curves*, i.e. how a neuron's firing rate
# varies with a behavioural variable, such as the animal's position, speed
# or head direction.
#
# Converting requires the optional ``pynapple`` dependency, which you can
# install with ``pip install "movement[pynapple]"``.
# See the "pynapple" dropdown in the :ref:`target-format-specific` section
# of the Input/Output guide for a summary.
#
# In this example, we compute some head-related variables, convert them to
# TsdFrames, and explore how they look. Finally, we use simulated spikes to
# show how the converted variables can be used to compute tuning curves.

# %%
# Load the data
# -------------
# We use a sample dataset of two mice exploring an octagonal arena,
# tracked with SLEAP. Time is in seconds, since the dataset comes with a
# frame rate (``fps``). This matters, because ``pynapple`` interprets all
# timestamps as seconds.

ds = sample_data.fetch_dataset("SLEAP_two-mice_octagon.analysis.h5")
print(f"fps: {ds.fps}")
ds

# %%
# Neural recordings are typically made in one animal per session,
# and :func:`~movement.io.to_pynapple.to_pynapple_tsdframe` converts one
# individual at a time. So we select mouse ``"1"``.

ds_mouse = ds.sel(individual="1")

# %%
# Clean the position data
# -----------------------
# We :ref:`remove low-confidence predictions
# <sphx_glr_examples_filter_and_interpolate.py>` and interpolate over gaps
# of up to 10 frames (0.2 s). Longer gaps are left as NaN; we'll come back
# to them later.

position = filter_by_confidence(
    ds_mouse.position, ds_mouse.confidence, threshold=0.6
)
position = interpolate_over_time(position, max_gap=10)
print(f"Remaining NaNs: {position.isnull().mean().item():.1%}")

# %%
# Compute head variables
# ----------------------
# Next we compute three variables that are often related to neural
# activity:
#
# - **Head position**: the midpoint between the ears.
# - **Head speed**: how fast the head position changes, see
#   :ref:`sphx_glr_examples_compute_kinematics.py`.
# - **Head direction**: the angle of the head's forward vector relative to
#   the x-axis, in radians, see
#   :ref:`sphx_glr_examples_compute_head_direction.py`.
#
# We rename the speed and direction DataArrays, because their names will
# become column names when we convert them (more on that below).

head_position = position.sel(keypoint=["EarLeft", "EarRight"]).mean("keypoint")
head_speed = compute_speed(head_position).rename("head_speed")
head_direction = compute_forward_vector_angle(
    position, left_keypoint="EarLeft", right_keypoint="EarRight"
).rename("head_direction")

# %%
# Each of these has fewer dimensions than ``position``: ``head_position``
# has ``time`` and ``space``, while ``head_speed`` and ``head_direction``
# only have ``time``.

for da in (position, head_position, head_speed, head_direction):
    print(f"{da.name}: {da.dims}")

# %%
# Convert to TsdFrames
# --------------------
# Let's start by converting the cleaned ``position`` of all keypoints with
# :func:`~movement.io.to_pynapple.to_pynapple_tsdframe`.

position_tsdframe = to_pynapple_tsdframe(position)
position_tsdframe

# %%
# The TsdFrame has one row per time point and one column per combination of
# keypoint and spatial coordinate, named ``<keypoint>_<space>``.
# Each column also carries *metadata*: the keypoint and space labels it was
# derived from, as well as the individual's name. Printing a TsdFrame only
# shows the first two metadata fields, but we can see all of them with:

position_tsdframe.metadata

# %%
# Columns can be selected by name, as in ``pandas``:

position_tsdframe[["Nose_x", "Nose_y"]]

# %%
# They can also be selected by metadata, e.g. all columns of the ``Nose``
# keypoint:

position_tsdframe[:, position_tsdframe.keypoint == "Nose"]

# %%
# Now let's convert the three head variables.

head_position_tsdframe = to_pynapple_tsdframe(head_position)
head_speed_tsdframe = to_pynapple_tsdframe(head_speed)
head_direction_tsdframe = to_pynapple_tsdframe(head_direction)

head_position_tsdframe

# %%
# Since ``head_position`` has no ``keypoint`` dimension, its columns are
# named after the spatial coordinates only: ``x`` and ``y``.
# ``head_speed`` and ``head_direction`` only have the ``time`` dimension,
# so each becomes a single column, named after the DataArray.
# This is why we renamed them: otherwise, the head direction column would
# have been named ``forward_vector_angle``.

head_direction_tsdframe

# %%
# The metadata shrinks along with the columns, but the individual's name is
# always kept:

for tsdframe in (
    head_position_tsdframe,
    head_speed_tsdframe,
    head_direction_tsdframe,
):
    print(tsdframe.metadata, end="\n\n")

# %%
# The converted variables can be plotted directly with ``matplotlib``.
# Here we use :meth:`~pynapple.TsdFrame.restrict` to select a 10-second
# interval (an :class:`~pynapple.IntervalSet`).

interval = nap.IntervalSet(start=60, end=70)

fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
axes[0].plot(head_position_tsdframe.restrict(interval), label=["x", "y"])
axes[0].set_ylabel("Head position (pixels)")
axes[0].legend(loc="upper right")
axes[1].plot(head_speed_tsdframe.restrict(interval), color="tab:green")
axes[1].set_ylabel("Head speed\n(pixels/s)")
axes[2].plot(head_direction_tsdframe.restrict(interval), color="tab:purple")
axes[2].set_ylabel("Head direction\n(rad)")
axes[2].set_xlabel("Time (s)")
fig.tight_layout()
plt.show()

# %%
# Save to disk
# ------------
# ``movement`` doesn't save TsdFrames itself, but ``pynapple`` does.
# :meth:`~pynapple.TsdFrame.save` writes a TsdFrame, including its column
# names and metadata, to a NumPy ``.npz`` file, which can be loaded back with
# :func:`~pynapple.io.misc.load_file`. Here we use a temporary directory;
# replace it with a path of your choice.

save_dir = Path(tempfile.mkdtemp())
head_direction_tsdframe.save(save_dir / "head_direction.npz")
nap.load_file(save_dir / "head_direction.npz")

# %%
# Compute a tuning curve
# ----------------------
# Head direction cells are neurons that fire when the animal's head points in
# a particular direction, their *preferred direction*. Their tuning curve,
# i.e. the firing rate as a function of head direction, peaks around that
# direction.
#
# This dataset doesn't include neural recordings, so we simulate the spikes of
# a head direction cell. Its firing rate peaks at a preferred direction of
# :math:`\pi/2` (90°), following a
# `von Mises <https://en.wikipedia.org/wiki/Von_Mises_distribution>`_
# shape. With real data, you would instead load your spike times into a
# :class:`~pynapple.TsGroup`.
#
# First, we need to deal with the remaining gaps in ``head_direction``.
# ``pynapple`` functions don't all handle NaNs, so we drop the time points
# with NaNs, using :meth:`~pynapple.TsdFrame.dropna`. This also excludes them
# from the TsdFrame's *time support*, i.e. the intervals over which the data
# is defined, so that later operations skip them.

head_direction_valid = head_direction_tsdframe.dropna()
print(f"Time points kept: {len(head_direction_valid)} / {len(ds.time)}")
head_direction_valid.time_support

# %%
# Now we simulate the spikes. In each frame, we draw the number of spikes
# from a Poisson distribution, with a rate that depends on the current head
# direction, and spread them uniformly within the frame.

PREFERRED_DIRECTION = np.pi / 2  # in radians
MAX_RATE = 40  # in Hz, at the preferred direction
BASELINE_RATE = 1  # in Hz, away from the preferred direction
CONCENTRATION = 4  # the higher, the narrower the tuning

direction = head_direction_valid.values[:, 0]
rate = BASELINE_RATE + MAX_RATE * np.exp(
    CONCENTRATION * (np.cos(direction - PREFERRED_DIRECTION) - 1)
)
frame_duration = 1 / ds.fps
rng = np.random.default_rng(seed=42)
n_spikes = rng.poisson(rate * frame_duration)
spike_times = np.repeat(head_direction_valid.times(), n_spikes)
spike_times += rng.uniform(0, frame_duration, size=n_spikes.sum())

spikes = nap.TsGroup({0: nap.Ts(np.sort(spike_times))})
spikes

# %%
# We pass the spikes and the head direction to
# :func:`~pynapple.process.tuning_curves.compute_tuning_curves`, which bins
# the head direction and computes the firing rate in each bin.
# Since head direction is an angle, we set the bins' ``range`` to
# :math:`[-\pi, \pi]`, regardless of which directions the mouse visited.

tuning_curves = nap.compute_tuning_curves(
    spikes, head_direction_valid, bins=36, range=(-np.pi, np.pi)
)
tuning_curves

# %%
# The result is an :class:`xarray.DataArray` with one tuning curve per unit
# (here, just one). Let's plot it on a polar axis, along with the preferred
# direction we used to simulate the spikes.

# sphinx_gallery_thumbnail_number = 2

tuning_curve = tuning_curves.sel(unit=0)
bin_centres = tuning_curve.head_direction.values

fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(4, 4))
ax.plot(
    np.append(bin_centres, bin_centres[0]),  # close the curve
    np.append(tuning_curve.values, tuning_curve.values[0]),
    color="tab:purple",
)
ax.axvline(PREFERRED_DIRECTION, color="black", linestyle="--")
ax.set_title("Simulated head direction cell\nfiring rate (Hz)")
fig.tight_layout()
plt.show()

# %%
# As expected, the tuning curve peaks around the preferred direction
# (dashed line). With real data, the same steps apply to any of the converted
# variables, e.g. ``head_position_tsdframe`` to compute the 2D tuning curves
# of place cells. For more, see ``pynapple``'s
# :doc:`tuning curves guide <pynapple:user_guide/06_tuning_curves>`.
