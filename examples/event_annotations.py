"""Annotate time with events of interest
========================================

Label timepoints with events of interest and select
subsets of data by event.
"""

# %%
# Overview
# --------
# In movement data analysis, it is often useful to label each
# timepoint with **events of interest** — contextual information
# about what is happening at that moment. This could be a
# behavioural state (e.g., "active" vs "inactive"), a discrete
# event (e.g., stimulus onset, trial boundary), or any other
# per-timepoint annotation relevant to the analysis.
#
# These annotations can be stored as `non-dimension coordinates
# <https://docs.xarray.dev/en/stable/user-guide/data-structures.html#coordinates>`_
# along the ``time`` dimension, allowing us to select data using
# standard ``xarray`` operations.
#
# Here we demonstrate this using behavioural states. We
# take a 1-hour recording of a mouse in its home cage, compute its
# speed, and segment the recording into active and inactive states
# based on a speed threshold.
#
# .. note::
#    This example is inspired by code written by
#    `Callum Marshall <https://www.keshavarzilab.com/callum>`_ from the
#    `Keshavarzi lab <https://www.keshavarzilab.com>`_ at the
#    University of Cambridge.

# %%
# Imports
# -------

import matplotlib.pyplot as plt
import numpy as np

from movement import sample_data
from movement.kinematics import compute_speed

# %%
# Load a sample dataset
# ---------------------
# This contains the DeepLabCut predictions for a single mouse tracked over
# one hour in its home cage. We select the first (and only)
# individual and the ``bodycenter`` keypoint to work with.
# Let's call the selected dataset ``ds_bc`` for brevity.

ds = sample_data.fetch_dataset(
    "DLC_smart-kage3_datetime-20240417T100006.predictions.h5"
)
ds_bc = ds.sel(individuals="individual_0", keypoints="bodycenter")
print(ds_bc)

# %%
# Compute and plot speed
# ----------------------
# We use :func:`movement.kinematics.compute_speed` to compute the
# instantaneous speed of the ``bodycenter`` keypoint across time.
# We store the result as a new variable in the same dataset.

ds_bc["speed"] = compute_speed(ds_bc.position)
print(ds_bc.speed)

# %%
# We define a small helper function to plot speed over time, since
# we will reuse it several times. It can optionally overlay a
# threshold line and shade active periods. We'll see how that's useful
# in a moment.


def plot_speed(speed_da, threshold=None, active=None, title="Speed"):
    """Plot a speed DataArray over time."""
    fig, ax = plt.subplots(figsize=(8, 3))
    time = speed_da.time.values
    speed_da.plot.line(x="time", ax=ax, linewidth=0.5, color="grey")

    if threshold is not None:
        ax.axhline(threshold, linestyle="--", color="tab:red")

    if active is not None:
        changes = np.diff(active.astype(int), prepend=~active[0])
        starts = np.where(changes != 0)[0]
        ends = np.append(starts[1:], len(time))
        for s, e in zip(starts, ends, strict=True):
            if active[s]:
                ax.axvspan(
                    time[s],
                    time[e - 1],
                    alpha=0.2,
                    color="tab:green",
                )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (pixels/s)")
    ax.set_title(title)
    ax.set_xlim(time[0], time[-1])

    fig.subplots_adjust(bottom=0.2)


# %%
# Let's plot the speed over time to get a sense of the data.
# Note that speed here is expressed in pixels per second,
# because the DeepLabCut predictions are given in pixel coordinates
# and the time unit is seconds (stored in ``ds.attrs['time_unit']``).

plot_speed(ds_bc.speed, title="Speed")

# %%
# Define active vs inactive states
# ---------------------------------
# We apply an arbitrary speed threshold to classify each timepoint.
# This produces a boolean array where ``True`` means active
# and ``False`` means inactive.

speed_threshold = 40  # pixels/s
is_active = ds_bc.speed.values > speed_threshold

# %%
# We can now plot the speed over time with shaded bands
# highlighting active periods.

# sphinx_gallery_thumbnail_number = 2

plot_speed(
    ds_bc.speed,
    threshold=speed_threshold,
    active=is_active,
    title="Active periods",
)

# %%
# Select data by state
# --------------------
# We attach the boolean array as a new coordinate along
# the ``time`` dimension using
# :meth:`xarray.Dataset.assign_coords`.
# In ``xarray`` terminology, this is a `non-dimension coordinate
# <https://docs.xarray.dev/en/stable/user-guide/data-structures.html#coordinates>`_:
# it does not define a new dimension, but rather
# annotates an existing one with additional labels.

ds_bc = ds_bc.assign_coords(active=("time", is_active))
print(ds_bc.coords["active"])

# %%
# With the ``active`` coordinate in place, we can select
# subsets of the dataset directly with :meth:`xarray.Dataset.sel`.

# Select only active timepoints
ds_active = ds_bc.sel(active=True)
print(f"Active frames: {ds_active.sizes['time']}")

# Select only inactive timepoints
ds_inactive = ds_bc.sel(active=False)
print(f"Inactive frames: {ds_inactive.sizes['time']}")


# %%
# Stack multiple event annotations
# --------------------------------
# Each non-dimension coordinate along ``time`` acts like an
# **event annotation layer**: a parallel track of labels attached to the
# same time axis. We can stack multiple boolean layers, each
# capturing a different binary aspect of the data.
#
# For example, suppose we also know whether a stimulus was
# present at each timepoint. We can add a second boolean
# coordinate alongside ``active``.

# Create a dummy "stimulus" annotation, "on" for 30% of frames
rng = np.random.default_rng(42)
is_stimulus = rng.random(ds_bc.sizes["time"]) > 0.7
ds_bc = ds_bc.assign_coords(stimulus=("time", is_stimulus))

# %%
# While :meth:`~xarray.Dataset.sel` works for selecting on a
# single non-dimension coordinate, it does not support selecting
# on multiple non-dimension coordinates along the same dimension
# at once. To combine conditions across layers, we use boolean
# indexing along ``time`` instead.

ds_active_stim = ds_bc.sel(time=ds_bc.active & ds_bc.stimulus)
print(f"Active + stimulus frames: {ds_active_stim.sizes['time']}")

# %%
# This pattern scales to any number of boolean annotation layers
# without changing the dataset's dimensionality.

# %%
# Beyond boolean annotations
# --------------------------
# Boolean coordinates work well when each annotation has exactly
# two levels. But what if there are more? For example, a
# behavioural classifier might distinguish between "inactive",
# "grooming", and "running".
#
# In that case, string labels are more appropriate. Here we
# recreate the same active/inactive annotation as a string
# coordinate to illustrate the pattern.

state_labels = np.where(is_active, "active", "inactive")
ds_bc = ds_bc.assign_coords(state=("time", state_labels))
print(ds_bc.coords["state"])

# %%
# Selection works the same way with :meth:`~xarray.Dataset.sel`.

for state_name in ["active", "inactive"]:
    ds_state = ds_bc.sel(state=state_name)
    print(f"{state_name} frames: {ds_state.sizes['time']}")

# %%
# .. tip::
#    To select multiple states at once (useful with more than two
#    categories), use :meth:`xarray.DataArray.isin`:
#
#    .. code-block:: python
#
#       ds_bc.sel(
#           time=ds_bc.state.isin(["grooming", "running"])
#       )
