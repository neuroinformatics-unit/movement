"""Annotate and load events with BORIS
======================================

Manually annotate gait phase events in BORIS and load them into a
``movement`` dataset as per-frame labels for data selection and visualisation.
"""
# %%
# Overview
# --------
# This example demonstrates how to label gait phase events using
# `BORIS (Behavioural Observation Research Interactive Software)
# <https://www.boris.unito.it/>`_ and load them into a
# ``movement``-compatible format for downstream analysis.
#
# Gait phase - whether each limb is in stance (in contact with the
# ground), swing (in the air), or unknown (e.g. when the limb is off-screen
# or occluded) - is an example of a case where we want to annotate events that
# are perhaps not easily defined by a simple threshold on a single variable,
# but rather require visual inspection of the video data. Such manual
# annotations can be used directly for analysis in small datasets,
# or as training data for supervised learning models.
#
# In this example, we will use the ``DLC_single-mouse_DBTravelator_3D``
# dataset. This contains 3D pose estimates of the limbs and body of a
# single mouse locomoting on a dual-belt travelator, transitioning from one
# belt onto a second faster belt.

# %%
# Imports
# -------
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pooch
from matplotlib.patches import Patch

from movement import sample_data
from movement.filtering import filter_by_confidence

# %%
# Load sample dataset and media
# -----------------------------
# First, let's load the 3D pose dataset and separately download one of
# the source videos from which it was derived using `pooch
# <https://github.com/fatiando/pooch>`_. We will use this video to manually
# annotate gait phase events in BORIS.

ds = sample_data.fetch_dataset(
    "DLC_single-mouse_DBTravelator_3D.predictions.h5"
)

# Download the video
vid_name = "single-mouse_DBTravelator_video.avi"
video_path = pooch.retrieve(
    url=(
        f"https://gin.g-node.org/neuroinformatics/movement-test-data/raw"
        f"/master/videos/{vid_name}"
    ),
    known_hash=None,
    fname=vid_name,
    path=Path.home() / ".movement/data/videos",
)

# %%
# The video will be saved to your local movement cache directory and the path
# stored in ``video_path``, e.g.:
#
# .. code::
#
#    C:/Users/<username>/.movement/data/videos/
#    single-mouse_DBTravelator_video.avi
#
# Make a note of this path as you will need it to open the video in BORIS.

# %%
# Annotate gait phase events in BORIS
# -----------------------------------
# The following steps describe how to annotate gait phase events for four
# limbs in the downloaded video, producing labels that we will later load
# into the associated dataset ``ds``. For a more complete guide, refer to
# the `BORIS user guide <https://www.boris.unito.it/user_guide/>`_.
#
# **Step 1: Install BORIS**
#
# Download and install BORIS from the
# `BORIS website <https://www.boris.unito.it/>`_.
# Full installation instructions can be found in the
# `BORIS installation guide <https://www.boris.unito.it/user_guide/install/>`_.
#
# .. note::
#
#    This example was created using BORIS **v9.8.5**. The steps described
#    here may differ slightly for other versions.
#
# **Step 2: Create a new project**
#
# Open BORIS and create a new project via **Project > New Project**.
# In the dialogue that appears:
#
# - Set a **Project name**, e.g. ``label_gait``.
# - Add a brief **Project description**.
# - Set **Project time format** to ``seconds``.
#
# **Step 3: Build the behaviour ethogram**
#
# Navigate to the **Ethogram** tab. An ethogram is a catalogue of all
# events (or behaviours) to be annotated, where each event is assigned a
# keyboard shortcut for fast labelling. Here, each event represents a gait
# phase state for a single limb, e.g. left forepaw stance.  For each
# combination of limb (``FL``, ``FR``, ``HL``, ``HR``) and phase (
# ``stance``, ``swing``, ``unknown``), add a new behaviour:
#
# 1. Click **Behaviour > Add new behaviour**.
# 2. Under **Behaviour type**, select **State event** - a state event has a
#    duration, defined by a start and end time, as opposed to a point event
#    which is instantaneous.
# 3. Assign a unique **Key**, e.g. ``q`` for ``FL_stance``.
# 4. Set the **Code**, e.g. ``FL_stance``.
# 5. Repeat until all 12 behaviours (4 limbs × 3 phases) are defined.
#
# .. image:: /_static/events_ethogram.png
#    :width: 600
#
# 6. Because gait phase states cannot co-occur within a limb, we need to set
#    exclusion criteria to define which behaviours are mutually exclusive.
#    Open the **Exclusion matrix** and tick all mutually exclusive pairs
#    (e.g. ``FL_stance`` with ``FL_swing``). With this configured, starting
#    a new state within a limb will automatically close the previous state for
#    the same limb.
#
# .. image:: /_static/events_exclusions.png
#    :width: 600
#
# **Step 4: Start an observation**
#
# Create a new observation via **Observations > New observation**:
#
# - Set an **Observation ID**, e.g. ``individual_0``.
# - Tick **Observation from media file(s)**.
# - Click **Add media > with absolute path** and navigate to the
#   video file downloaded above at ``video_path``.
#
# - Click **Start**.
#
# **Step 5: Annotate events**
#
# - Use the ``←`` and ``→`` arrow keys or the upper panel buttons to step
#   through the video frame-by-frame.
# - Press the specified keyboard shortcut for a behaviour to mark its start
#   at the current frame.
# - Once the full video is annotated, close any remaining open state events
#   (i.e. the final behaviour for each limb, which has no subsequent behaviour
#   to trigger an automatic stop) via **Observations > Fix unpaired events**.
#   When prompted, enter a time near the end of the video. Note: BORIS staggers
#   each closing event by +1 ms to ensure unique timestamps, so choosing a
#   time too close to the end may place some events beyond the video duration.
#   For this video (total duration: 1.692 s; limbs: 4), we use 1.688 s,
#   giving closing events at 1.688, 1.689, 1.690, and 1.691 s.
# - BORIS sets frame indexes to NA for automatically generated stop events when
#   they do not correspond to a frame boundary. Run **Observations > Add
#   frame indexes** before exporting to populate these.
#
# .. image:: /_static/events_observation.png
#    :width: 600
#
# **Step 6: Export the event data**
#
# Export the annotations via
# **Observations > Export events > Aggregated events** and save as a CSV.

# %%
# Import BORIS event data into movement
# -------------------------------------
# We will now load the exported CSV file into a pandas DataFrame. If you
# have created your own annotations following
# the steps above, replace ``gait_events_path`` with the path to your
# exported CSV file, e.g.:
#
# .. code-block:: python
#
#     gait_events_path = "/your/path/to/aggregated.csv"
#
# Load the event data from CSV into a pandas Dataframe using
# :func:`pandas.read_csv`.

# sphinx_gallery_start_ignore
import pathlib  # noqa: E402
import tempfile  # noqa: E402

csv_data = """\
Behavior,Start (s),Stop (s),Image index start,Image index stop
FL_unknown,0.000,0.627,0,155
HL_unknown,0.000,0.716,0,177
FR_unknown,0.000,0.525,0,130
HR_unknown,0.000,0.635,0,157
FR_stance,0.526,0.627,130,155
FL_stance,0.628,0.712,155,176
FR_swing,0.628,0.716,155,177
HR_stance,0.636,0.712,157,176
FL_swing,0.713,0.809,176,200
HR_swing,0.713,0.793,176,196
HL_stance,0.717,0.793,177,196
FR_stance,0.717,0.798,177,197
HL_swing,0.794,0.882,196,218
HR_stance,0.794,0.869,196,215
FR_swing,0.798,0.882,197,218
FL_stance,0.810,0.878,200,217
HR_swing,0.870,0.950,215,235
FL_swing,0.879,0.959,217,237
HL_swing,0.882,0.883,218,218
HL_stance,0.883,0.942,218,233
FR_stance,0.883,0.946,218,234
HL_swing,0.943,1.023,233,253
FR_swing,0.947,1.031,234,255
HR_stance,0.951,1.019,235,252
FL_stance,0.960,1.023,237,253
HR_swing,1.020,1.104,252,273
FL_swing,1.024,1.104,253,273
HL_stance,1.024,1.092,253,270
FR_stance,1.032,1.100,255,272
HL_swing,1.093,1.173,270,290
FR_swing,1.101,1.177,272,291
FL_stance,1.105,1.169,273,289
HR_stance,1.105,1.173,273,290
FL_swing,1.170,1.250,289,309
HL_stance,1.174,1.242,290,307
HR_swing,1.174,1.250,290,309
FR_stance,1.178,1.250,291,309
HL_swing,1.243,1.335,307,330
FL_stance,1.251,1.339,309,331
FR_swing,1.251,1.319,309,326
HR_stance,1.251,1.339,309,331
FR_stance,1.320,1.477,326,365
HL_stance,1.336,1.452,330,359
FL_swing,1.340,1.444,331,357
HR_swing,1.340,1.416,331,350
HR_stance,1.417,1.546,350,382
FL_stance,1.445,1.546,357,382
HL_swing,1.453,1.546,359,382
FR_swing,1.478,1.533,365,379
FR_stance,1.534,1.542,379,381
FR_unknown,1.543,1.689,381,417
FL_swing,1.547,1.562,382,386
HL_stance,1.547,1.614,382,399
HR_swing,1.547,1.590,382,393
FL_unknown,1.563,1.688,386,417
HR_unknown,1.591,1.691,393,417
HL_unknown,1.615,1.690,399,417
"""

gait_events_path = pathlib.Path(tempfile.mkdtemp()) / "aggregated.csv"
gait_events_path.write_text(csv_data)
# sphinx_gallery_end_ignore

df = pd.read_csv(gait_events_path)

print(
    f"Key columns from the BORIS file:\n"
    f"{
        df.loc[
            :5,
            [
                'Behavior',
                'Start (s)',
                'Stop (s)',
                'Image index start',
                'Image index stop',
            ],
        ]
    }"
)

# %%
# Each row corresponds to a single annotated event, with columns for the
# behaviour code, start and stop times (in seconds), and start and stop
# frame indices.
#
# To attach these labels to ``ds`` as per-frame gait phase labels, we first
# reformat the event data, parsing the behaviour codes into separate columns
# for limb, state, and frame indices.

limbs = ["FL", "FR", "HL", "HR"]

events = (
    df["Behavior"]
    .str.split("_", expand=True)
    .rename(columns={0: "limb", 1: "state"})
)
events["start_frame"] = df["Image index start"]
events["stop_frame"] = df["Image index stop"]

# Snap the final stop frame per limb to n_frames so the last frame is included
events.loc[events.groupby("limb")["stop_frame"].idxmax(), "stop_frame"] = (
    ds.time.size
)

print(f"Parsed events:\n{events.head()}")

# %%
# Each row in ``events`` defines a phase label over a range of frames rather
# than for a single frame. We therefore expand this into a per-frame
# representation, initialising a 2-D array of shape ``(n_frames, n_limbs)``
# with ``NaN`` and filling each interval with the corresponding phase label.

phase_data = np.full((ds.time.size, len(limbs)), np.nan, dtype=object)

for limb_idx, limb in enumerate(limbs):
    limb_events = events[events["limb"] == limb]
    lengths = (limb_events["stop_frame"] - limb_events["start_frame"]).values
    states = limb_events["state"].values

    phase_data[:, limb_idx] = np.repeat(states, lengths)

print(f"10 sample frames from `phase_data`:\n{phase_data[210:220]}")

# %%
# Before attaching the labels, let's inspect ``ds``.

print(ds)

# %%
# The dataset currently has four dimension coordinates - ``time``,
# ``space``, ``keypoints``, and ``individuals`` - but no information about
# gait phase.

# %%
# In xarray, additional per-timepoint information can be attached alongside
# the dimension coordinates as `non-dimension coordinates
# <https://docs.xarray.dev/en/stable/user-guide/data-structures.html
# #coordinates>`_. We use :meth:`xarray.Dataset.assign_coords` to attach
# the per-frame gait phase labels as non-dimension coordinates on the
# ``time`` dimension, one per limb, following the naming convention
# ``gait_<limb>``. These non-dimension coordinates can be thought of as
# metadata associated with each point along the ``time`` dimension.

for limb_idx, limb in enumerate(limbs):
    ds = ds.assign_coords({f"gait_{limb}": ("time", phase_data[:, limb_idx])})

print(ds)

# %%
# We now have four new coordinates on the ``time`` dimension - one per limb
# - each containing the gait phase label for every frame.

# %%
# Select data by gait phase
# -------------------------
# With the gait phase coordinates in place, we can now select subsets of
# ``ds`` by gait using :meth:`~xarray.DataArray.sel`. For example, to select
# all timepoints where the front-right paw is in stance:

ds_fr_stance = ds.position.sel(time=ds.gait_FR == "stance")
print(ds_fr_stance)

# %%
# As a sanity check, we can now visualise the z-position (height) of the
# mouse's right forepaw, colour-coded by gait phase. We would expect the
# paw to be elevated during swing and near the belt surface (z=0) during
# stance.

# First, we filter the pose data by confidence
ds["position"] = filter_by_confidence(
    ds.position,
    ds.confidence,
    threshold=0.9,
)

# Select the z-position of the right forepaw toe
fr_z = ds.position.sel(keypoints="ForepawToeR", space="z")

# Select frames by gait phase using the gait_FR coordinate
fr_swing = fr_z.sel(time=ds.gait_FR == "swing")
fr_stance = fr_z.sel(time=ds.gait_FR == "stance")

fig, ax = plt.subplots(figsize=(8, 3))
ax.scatter(fr_swing.time, fr_swing.values, s=8, label="Swing")
ax.scatter(fr_stance.time, fr_stance.values, s=8, label="Stance")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Z position (mm)")
ax.legend()
plt.tight_layout()
plt.show()

# %%
# Indeed, we see a clear separation between swing and stance phases,
# with the paw elevated during swing and near the belt surface during stance.
# Note, that the position of the toe at the end of stance phase can be greater
# in z than time points in swing phase, possibly reflecting subtle tracking
# inconsistencies when the foot is in different positions. This illustrates
# why a simple threshold-based approach may not always be sufficient for
# tasks like gait phase detection.

# %%
# Next, to select frames where multiple conditions hold simultaneously, we can
# also combine any number of boolean masks across limb coordinates using ``&``.

# Create a boolean mask for each limb in stance
fl_stance = ds["gait_FL"] == "stance"
fr_stance = ds["gait_FR"] == "stance"
hl_stance = ds["gait_HL"] == "stance"
hr_stance = ds["gait_HR"] == "stance"

# Select frames where all four limbs are simultaneously in stance
all_stance = ds.position.sel(
    time=fl_stance & fr_stance & hl_stance & hr_stance
)
print(all_stance)

# %%
# Here we see that there are very few frames in which all four limbs are
# simultaneously in stance, as expected during continuous locomotion.

# %%
# We can use the same approach to identify frames in which diagonal limb
# pairs are simultaneously in stance. When mice locomote at intermediate
# speeds, they typically display a trotting gait, in which diagonal limb
# pairs (e.g. front-left and hind-right) move in synchrony. We might
# therefore expect stance phases to overlap within each diagonal pair in our
# dataset. Here we plot the stance phase of each limb, highlighting frames
# where both limbs of a diagonal pair are simultaneously in stance.

fig, ax = plt.subplots(figsize=(10, 3))

limbs_ordered = ["HL", "FL", "FR", "HR"]
diagonal_colors = {
    "FL": "steelblue",
    "HR": "steelblue",
    "FR": "tomato",
    "HL": "tomato",
}

fl_hr_stance = (ds["gait_FL"] == "stance") & (ds["gait_HR"] == "stance")
fr_hl_stance = (ds["gait_FR"] == "stance") & (ds["gait_HL"] == "stance")

frame_duration = 1 / ds.fps
time_vals = ds.time.values

for i, limb in enumerate(limbs_ordered):
    # Pick a diagonal pair mask
    diagonal_mask = fl_hr_stance if limb in {"FL", "HR"} else fr_hl_stance
    # Find the single limb stance phase frames
    stance_mask = ds[f"gait_{limb}"] == "stance"

    # Frames where diagonal partner is also in stance
    for t in ds.time.sel(time=stance_mask & diagonal_mask).values:
        ax.barh(
            i,
            frame_duration,
            left=t,
            height=0.9,
            color=diagonal_colors[limb],
            edgecolor="none",
        )

    # Remaining stance frames
    for t in ds.time.sel(time=stance_mask & ~diagonal_mask).values:
        ax.barh(
            i,
            frame_duration,
            left=t,
            height=0.9,
            color="lightgrey",
            edgecolor="none",
        )

ax.set_yticks(range(len(limbs_ordered)))
ax.set_yticklabels(limbs_ordered)
ax.set_xlabel("Time (s)")
ax.invert_yaxis()

# Plot legend
handles = [
    Patch(color="steelblue", label="FL & HR stance"),
    Patch(color="tomato", label="FR & HL stance"),
    Patch(color="lightgrey", label="Stance"),
]
ax.legend(handles=handles, loc="upper right")

plt.tight_layout()
plt.show()
# sphinx_gallery_thumbnail_number = 2

# %%
# As expected, stance phases largely overlap within diagonal pairs - when
# the front-left limb is in stance, the hind-right limb tends to be too,
# and vice versa for front-right and hind-left, confirming that the mouse is
# locomoting with a trotting gait.

# %%
# :meth:`~xarray.DataArray.isin` can also be used to select frames where
# a limb is in any one of multiple phases at once. For example, to retain
# only frames where the front-right limb phase was confidently identified
# (i.e. excluding ``"unknown"``):

fr_visible = ds.position.sel(time=ds.gait_FR.isin(["stance", "swing"]))
print(fr_visible)


# %%
# Segment data by stride
# ----------------------
# Building on the per-frame gait phase labels we have just attached, we can
# perform a second 'level' of labelling frames by grouping consecutive
# stance-swing cycles into individual strides. We define a stride as one
# complete stance-swing sequence for a chosen reference limb, and assign a
# stride index to each frame as an additional non-dimension coordinate on the
# ``time`` dimension. Frames outside of a complete stride cycle are left as
# ``NaN``.

# %%
# First, we define a helper function to find the start and end frame indices
# of contiguous blocks of a given phase label in a coordinate array.


def find_phase_blocks(phase_coord, phase_label):
    """Find start and end frame indices of contiguous blocks of a given phase.

    Parameters
    ----------
    phase_coord : np.ndarray
        Array of phase labels for each frame.
    phase_label : str
        The phase label to find blocks of.

    Returns
    -------
    list of tuple
        List of (start, end) frame index pairs for each contiguous block.

    """
    is_phase = (phase_coord == phase_label).astype(int)
    transitions = np.diff(is_phase, prepend=0, append=0)
    starts = np.where(transitions == 1)[0]
    ends = np.where(transitions == -1)[0]
    return list(zip(starts, ends, strict=True))


# %%
# Using the ``gait_FR`` coordinate already attached to the ``time``
# dimension, we find all contiguous stance and swing blocks for the
# front-right limb, then pair each stance block with the immediately
# following swing block to form a complete stride.

# Get the gait phase labels for the front-right limb
gait_coord = ds.gait_FR.values

# Find contiguous blocks of stance and swing phases
stance_blocks = find_phase_blocks(gait_coord, "stance")
swing_blocks = find_phase_blocks(gait_coord, "swing")

stride_data = np.full(ds.time.size, np.nan)
stride_idx = 0
for stance_start, stance_end in stance_blocks:
    # Find the swing block that starts immediately after this stance block
    following_swings = [(s, e) for s, e in swing_blocks if s == stance_end]
    if following_swings:
        # Label all frames from stance onset to swing offset with the stride
        # index
        swing_start, swing_end = np.squeeze(following_swings)
        stride_data[stance_start:swing_end] = stride_idx
        stride_idx += 1

# Attach the stride index as a non-dimension coordinate on the time dimension
ds = ds.assign_coords(stride_FR=("time", stride_data))

print(ds.isel(time=np.arange(150, 350)))

# %%
# We can now select data by stride index in the same way as with gait phase.
# To visualise the segmentation, we plot the z-position of the front-right limb
# for each stride as a separate coloured line, with stance and swing
# phases shown by shaded boxes. Dashed vertical lines mark the boundaries
# between consecutive strides.

# Plot with stance and swing phases highlighted
fr = ds.position.sel(keypoints="ForepawToeR", space="z")

fig, ax = plt.subplots(figsize=(8, 3))

n_strides = int(np.nanmax(fr.stride_FR.values)) + 1
# Plot each stride trajectory
for i in range(n_strides):
    stride = fr.sel(time=fr.stride_FR == i)
    ax.plot(stride.time, stride.values.squeeze(), linewidth=1.5, zorder=100)
    ax.scatter(stride.time, stride.values.squeeze(), s=8, zorder=100)

# Plot shaded boxes for stance and swing phases, and vertical lines at
# stride boundaries
for phase, color in [("stance", "white"), ("swing", "lightgrey")]:
    blocks = find_phase_blocks(fr["gait_FR"].values, phase)
    for j, (s, e) in enumerate(blocks):
        # Plot shaded box for this phase block
        ax.axvspan(
            fr.time.values[s],
            fr.time.values[e - 1],
            alpha=0.4,
            zorder=1,
            color=color,
            label=phase if j == 0 else None,
        )
        # plot boundary line between consecutive strides
        if phase == "swing":
            ax.axvline(
                fr.time.values[e - 1],
                color="grey",
                linestyle="--",
                alpha=0.5,
                zorder=200,
            )

ax.set_xlabel("Time (s)")
ax.set_ylabel("Z position (mm)")
ax.legend()
plt.tight_layout()
plt.show()

# %%
# We can select data by stride index to compare the paw trajectories across
# strides. Here we align each front-right limb stride by its end frame and
# plot the mean trajectory alongside individual strides.

# Extract each stride into a list
strides = []
for i in range(n_strides):
    stride = fr.sel(time=fr.stride_FR == i).values.squeeze()
    strides.append(stride)

# Align by last frame of each stride
last_indices = [len(s) - 1 for s in strides]
max_stride_len = max(last_indices)
stride_array = np.full((n_strides, max_stride_len + 1), np.nan)
for i, (s, last) in enumerate(zip(strides, last_indices, strict=True)):
    start = max_stride_len - last
    stride_array[i, start : start + len(s)] = s

frame_indices = np.arange(stride_array.shape[1]) - max_stride_len
mean_stride = np.nanmean(stride_array, axis=0)

fig, ax = plt.subplots(figsize=(4, 3))

for i in range(n_strides):
    ax.plot(
        frame_indices,
        stride_array[i],
        color="steelblue",
        alpha=0.3,
        linewidth=0.8,
    )

ax.plot(
    frame_indices,
    mean_stride,
    color="steelblue",
    linewidth=2,
    label="Mean stride",
)

ax.axvline(0, color="k", linestyle="--", linewidth=0.8, label="Stride end")
ax.set_xlabel("Frame relative to stride end")
ax.set_ylabel("Z position (mm)")
ax.legend()
plt.tight_layout()
plt.show()
