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
# can read NWB files out of the box. In this example we take a
# ``movement`` dataset, save it as an NWB file, and load it back with
# pynapple.

# %%
# Imports
# -------
import tempfile
from pathlib import Path

import pynapple as nap
from pynwb import NWBHDF5IO

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
# Save the dataset to an NWB file
# -------------------------------
# :func:`movement.io.save_poses.to_nwb_file` converts a ``movement``
# dataset into one or more :class:`pynwb.file.NWBFile` objects (one per
# individual), following the ``ndx-pose`` extension. Because our dataset
# has a single individual, a single ``NWBFile`` object is returned.
#
# The ``NWBFile`` object lives in memory, so we write it to disk with
# :class:`pynwb.NWBHDF5IO`. Here we use a temporary directory, but you
# would typically choose a permanent location.

nwb_file = save_poses.to_nwb_file(ds)

nwb_path = Path(tempfile.mkdtemp()) / "wasp.nwb"
with NWBHDF5IO(nwb_path, mode="w") as io:
    io.write(nwb_file)

print(f"Saved NWB file to: {nwb_path}")

# %%
# Load the NWB file with pynapple
# -------------------------------
# :func:`pynapple.load_file` reads the NWB file and returns an object
# that behaves like a dictionary. Each keypoint is exposed as a
# :class:`pynapple.TsdFrame` (a time-indexed table) with ``x`` and ``y``
# columns.

data = nap.load_file(str(nwb_path))
print(data)

# %%
# We can access the pose track for a single keypoint by name. The
# resulting ``TsdFrame`` is indexed by time (in seconds) and pynapple
# has inferred the sampling rate from the timestamps.

head = data["head"]
print(head)

# %%
# Use the data with pynapple
# --------------------------
# Once the data is a pynapple object, we can use pynapple's API. For
# example, we can define an :class:`pynapple.IntervalSet` and restrict
# the pose track to that time window.

interval = nap.IntervalSet(start=0, end=5)  # seconds
head_first_5s = head.restrict(interval)
print(head_first_5s)

# %%
# .. note::
#    Two things are worth keeping in mind when going through NWB:
#
#    - **Confidence scores are not carried over.** ``movement`` stores
#      per-keypoint confidence values, but pynapple only reads the
#      ``x``/``y`` position data from the NWB file.
#    - **Multiple individuals are split across files.** The NWB format
#      stores one individual per file, so
#      :func:`~movement.io.save_poses.to_nwb_file` returns a list of
#      ``NWBFile`` objects for multi-individual datasets. You would write
#      and load each one separately, and the individual's identity is
#      held in the file rather than in the pynapple object.
