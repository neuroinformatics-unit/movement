"""Convert to pynapple via NWB
=============================

Export a ``movement`` poses dataset to NWB and load it with pynapple.
"""

# %%
# Motivation
# ----------
# `pynapple <https://pynapple.org/>`_ is a lightweight Python package for
# analysing time series data, with a focus on neuroscience. It provides
# convenient objects for handling timestamps, epochs and time-varying
# variables (such as pose tracks).
#
# ``movement`` and pynapple can talk to each other through the
# `Neurodata Without Borders (NWB) <https://www.nwb.org/>`_ format:
# ``movement`` can export poses to NWB (using the
# `ndx-pose <https://github.com/rly/ndx-pose>`_ extension), and pynapple
# can read NWB data out of the box. In this example we take a
# ``movement`` dataset, convert it to NWB, and load it into pynapple.

# %%
# Imports
# -------
import numpy as np
import pynapple as nap

from movement import sample_data
from movement.io import save_poses

# %%
# Load a movement dataset
# -----------------------
# We use a sample dataset with a single individual (a wasp) tracked with
# two keypoints. Using a single-individual dataset keeps the mapping to
# pynapple objects straightforward (see the note at the end of this
# example for the multi-individual case).

ds = sample_data.fetch_dataset("DLC_single-wasp.predictions.h5")

print(ds)
print("-----------------------------")
print(f"Individuals: {ds.individual.values}")
print(f"Keypoints: {ds.keypoint.values}")

# %%
# Convert the dataset to NWB
# --------------------------
# :func:`movement.io.save_poses.to_nwb_file` converts a ``movement``
# dataset into one or more :class:`pynwb.file.NWBFile` objects (one per
# individual), following the ``ndx-pose`` extension. Because our dataset
# has a single individual, a single ``NWBFile`` object is returned.
#
# The ``NWBFile`` object lives in memory, so we can hand it straight to
# pynapple without writing anything to disk.

nwb_file = save_poses.to_nwb_file(ds)

# %%
# The easy path: load positions with pynapple
# -------------------------------------------
# :class:`pynapple.NWBFile` wraps an ``NWBFile`` object and exposes its
# contents like a dictionary. Each keypoint becomes a
# :class:`pynapple.TsdFrame` (a time-indexed table) with ``x`` and ``y``
# columns.

nap_file = nap.NWBFile(nwb_file)
print(nap_file)

# %%
# We can access the pose track for a single keypoint by name. The
# resulting ``TsdFrame`` is indexed by time (in seconds) and pynapple
# has inferred the sampling rate from the timestamps.

head = nap_file["head"]
print(head)

# %%
# Once the data is a pynapple object, we can use pynapple's API. For
# example, we can define an :class:`pynapple.IntervalSet` and restrict
# the pose track to that time window.

interval = nap.IntervalSet(start=0, end=5)  # seconds
print(head.restrict(interval))

# %%
# The complete path: recover confidence via pynwb
# -----------------------------------------------
# :class:`pynapple.NWBFile` only reads the ``x``/``y`` position data. The
# ``ndx-pose`` extension also stores a per-keypoint ``confidence`` score,
# which we can reach through the ``NWBFile`` object directly. The pose
# data lives in the ``"behavior"`` processing module, inside a
# ``PoseEstimation`` container that holds one series per keypoint.

pose_estimation = nwb_file.processing["behavior"]["PoseEstimation"]
head_series = pose_estimation.pose_estimation_series["head"]

print(f"position shape: {head_series.data.shape}")  # (time, space)
print(f"confidence shape: {head_series.confidence.shape}")  # (time,)

# %%
# We can build a richer :class:`pynapple.TsdFrame` that carries the
# confidence score alongside the ``x`` and ``y`` coordinates, by stacking
# the two arrays into a single table.

head_with_conf = nap.TsdFrame(
    t=head_series.timestamps[:],
    d=np.column_stack([head_series.data[:], head_series.confidence[:]]),
    columns=["x", "y", "confidence"],
)
print(head_with_conf)

# %%
# .. note::
#    To share or archive the data, you will want to write the NWB file to
#    disk. Use :class:`pynwb.NWBHDF5IO`::
#
#        from pynwb import NWBHDF5IO
#
#        with NWBHDF5IO("wasp.nwb", mode="w") as io:
#            io.write(nwb_file)
#
#    You can then load it back with :func:`pynapple.load_file`, which
#    opens the file and wraps it in a :class:`pynapple.NWBFile` for you.

# %%
# .. note::
#    **Multiple individuals are split across files.** The NWB format
#    stores one individual per file, so
#    :func:`~movement.io.save_poses.to_nwb_file` returns a *list* of
#    ``NWBFile`` objects for multi-individual datasets. You would wrap
#    (or write) each one separately, and the individual's identity is
#    held in the file rather than in the pynapple object.
