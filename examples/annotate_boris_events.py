"""Label and load events with BORIS
===================================

Label the gait phase of four limbs in BORIS and load into movement
compatible format.
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
# or occluded) - is an example of a case where we want to label events that
# are perhaps not easily defined by a simple threshold on a single variable,
# but rather require visual inspection of the video data. Such labels can be
# used directly for analysis in small datasets, or as training data for
# supervised learning models.
#
# In this example, we will use the ``DLC_single-mouse_DBTravelator_3D``
# dataset, which contains 3D pose estimates of four limbs and other body
# points for a single mouse locomoting on a dual-belt travelator, where it
# runs along one belt and onto a second faster belt.

# %%
# Imports
# -------
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from movement import sample_data

# %%
# Load sample dataset and media
# -----------------------------
# First, let's load the dataset and associated video file for visual
# inspection and event labelling in BORIS.

ds = sample_data.fetch_dataset(
    "DLC_single-mouse_DBTravelator_3D.predictions.h5"
)

# The associated 2D dataset includes the side camera video, which we will
# use for frame-by-frame event labelling in BORIS.
_ = sample_data.fetch_dataset(
    "DLC_single-mouse_DBTravelator_2D.predictions.h5", with_video=True
)

# %%
# Label gait phase events in BORIS
# --------------------------------
# The following steps describe how to label gait phase events for four limbs
# in the ``DLC_single-mouse_DBTravelator_3D`` dataset. For a more complete
# guide covering other use cases, refer to the
# `BORIS user guide <https://www.boris.unito.it/user_guide/>`_.
#
# **Step 1: Install BORIS**
#
# Download and install BORIS from the
# `BORIS website <https://www.boris.unito.it/>`_.
# Full installation instructions can be found in the
# `BORIS installation guide <https://www.boris.unito.it/user_guide/install/>`_.
#
# **Step 2: Create a new project**
#
# Open BORIS and create a new project via **Project > New Project**.
# In the dialogue that appears:
#
# - Set a **Project name**, e.g. ``label_gait``.
# - Add a brief **Project description**.
# - Set **Project time format** to ``Seconds``.
#
# **Step 3: Build the behaviour ethogram**
#
# Navigate to the **Ethogram** tab. An ethogram is a catalogue of all
# behaviours to be annotated, where each behaviour is assigned a keyboard
# shortcut for fast labelling. For each combination of limb
# (``FL``, ``FR``, ``HL``, ``HR``) and phase (``stance``, ``swing``,
# ``unknown``), add a new behaviour:
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
# 6. Because gait phase states cannot co-occur within a limb, open the
#    **Exclusion matrix** and tick all mutually exclusive pairs
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
# - Click **Add media > with absolute path** and navigate to the video
#   file fetched above with :func:`sample_data.fetch_dataset
#   <movement.sample_data.fetch_dataset>`::
#
#       C:/Users/<username>/.movement/data/videos/single
#       -mouse_DBTravelator_video.avi
#
# - Click **Start**.
#
# **Step 5: Label events**
#
# - Use the ``←`` and ``→`` arrow keys or the upper panel buttons to step
#   through the video frame-by-frame.
# - Press the selected keyboard shortcut for a behaviour to mark its start
#   at the current frame.
# - Once the full video is labelled, close any remaining open state events
#   (i.e. the final behaviour for each limb, which has no subsequent behaviour
#   to trigger an automatic stop) via **Observations > Fix unpaired events**.
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
# We now load the exported CSV file into a pandas DataFrame. If you have
# created your own labels following the steps above, replace
# ``gait_labels_path`` with the path to your exported CSV file, e.g.:
#
# .. code-block:: python
#
#     gait_labels_path = "/your/path/to/aggregated.csv"
#
# The CSV exported from BORIS contains one row per labelled event, with
# columns including the behaviour code, start time, and stop time (in seconds).

# sphinx_gallery_start_ignore
import pathlib  # noqa: E402

csv_data = """\
Behavior,Start (s),Stop (s)
FL_unknown,0.000,0.627
HL_unknown,0.000,0.716
FR_unknown,0.000,0.525
HR_unknown,0.000,0.635
FR_stance,0.526,0.627
FL_stance,0.628,0.712
FR_swing,0.628,0.716
HR_stance,0.636,0.712
FL_swing,0.713,0.809
HR_swing,0.713,0.793
HL_stance,0.717,0.793
FR_stance,0.717,0.798
HL_swing,0.794,0.882
HR_stance,0.794,0.869
FR_swing,0.798,0.882
FL_stance,0.810,0.878
HR_swing,0.870,0.950
FL_swing,0.879,0.959
HL_stance,0.883,0.938
FR_stance,0.883,0.946
HL_swing,0.939,1.023
FR_swing,0.947,1.031
HR_stance,0.951,1.019
FL_stance,0.960,1.023
HR_swing,1.020,1.104
FL_swing,1.024,1.104
HL_stance,1.024,1.092
FR_stance,1.032,1.100
HL_swing,1.093,1.173
FR_swing,1.101,1.177
FL_stance,1.105,1.169
HR_stance,1.105,1.173
FL_swing,1.170,1.250
HL_stance,1.174,1.242
HR_swing,1.174,1.250
FR_stance,1.178,1.250
HL_swing,1.243,1.335
FL_stance,1.251,1.339
FR_swing,1.251,1.319
HR_stance,1.251,1.339
FR_stance,1.320,1.477
HL_stance,1.336,1.452
FL_swing,1.340,1.444
HR_swing,1.340,1.416
HR_stance,1.417,1.546
FL_stance,1.445,1.546
HL_swing,1.453,1.546
FR_swing,1.478,1.533
FR_stance,1.534,1.542
FR_unknown,1.543,1.689
FL_swing,1.547,1.562
HL_stance,1.547,1.614
HR_swing,1.547,1.590
FL_unknown,1.563,1.688
HR_unknown,1.591,1.691
HL_unknown,1.615,1.690
"""
gait_labels_path = pathlib.Path("boris_gait_labels.csv")
gait_labels_path.write_text(csv_data)
# sphinx_gallery_end_ignore

df = pd.read_csv(gait_labels_path)

print(
    f"Key columns from the gait labels:\n"
    f"{df.loc[:5, ['Behavior', 'Start (s)', 'Stop (s)']]}"
)

# %%
# Let's now transform the event data into a format compatible with movement.
# We first parse the behaviour codes into a pandas DataFrame with separate
# columns for limb, state, and frame indices.
#
# BORIS records event times in seconds. We convert these to frame indices by
# multiplying by the frame rate and rounding to the nearest integer.

fps = ds.fps
n_frames = ds.time.size
limbs = ["FL", "FR", "HL", "HR"]

# Parse behaviour codes into limb, state, start frame, and stop frame columns.
events = (
    df["Behavior"]
    .str.split("_", expand=True)
    .rename(columns={0: "limb", 1: "state"})
)
events["start_frame"] = round(df["Start (s)"] * fps).astype(int)
events["stop_frame"] = round((df["Stop (s)"] + 0.001) * fps).astype(int)

# BORIS subtracts 1 ms from the final stop time for each limb when running
# "Fix unpaired events", which can round the stop frame down by one. We
# correct this by snapping the final stop frame for each limb to n_frames.
for limb in limbs:
    last_idx = events[events["limb"] == limb]["stop_frame"].idxmax()
    events.loc[last_idx, "stop_frame"] = n_frames

# %%
# In xarray, we can attach additional per-timepoint information to a dataset
# as `non-dimension coordinates
# <https://docs.xarray.dev/en/stable/user-guide/data-structures.html
# #coordinates>`_ along the ``time`` dimension, effectively labelling each
# frame with contextual information. Here we will attach the gait phase of each
# limb as a separate coordinate. To prepare these labels, we first construct
# a 2-D array of shape ``(n_frames, n_limbs)`` initialised to ``NaN``, then
# populate each element with the corresponding phase label for that frame.

phase_data = np.full((ds.time.size, len(limbs)), np.nan, dtype=object)

for limb_idx, limb in enumerate(limbs):
    limb_events = events[events["limb"] == limb].sort_values("start_frame")
    for _, row in limb_events.iterrows():
        phase_data[row["start_frame"] : row["stop_frame"], limb_idx] = row[
            "state"
        ]

# %%
# We add the gait phase for each limb as a non-dimension coordinate on the
# ``time`` dimension using :meth:`xarray.Dataset.assign_coords`, following
# the naming convention ``gait_<limb>``. This will allow us to filter any
# data variable in the dataset by gait phase using
# :meth:`xarray.DataArray.sel`.

for limb_idx, limb in enumerate(limbs):
    ds = ds.assign_coords({f"gait_{limb}": ("time", phase_data[:, limb_idx])})

print(ds)

# %%
# We now have four new coordinates on the ``time`` dimension - one per limb
# - each containing the gait phase label for every frame.

# %%
# Select data by gait phase
# -------------------------
# With the gait phase coordinates in place, we can select subsets of the
# dataset directly with :meth:`~xarray.DataArray.sel`. For example, all
# timepoints where the front-right paw is in stance:

ds_fr_stance = ds.position.sel(time=ds.gait_FR == "stance")
print(ds_fr_stance)

# %%
# As a sanity check, we can visualise the z-position (height) of the right
# forepaw toe, colour-coded by gait phase. We would expect the paw to be
# elevated during swing and near the belt surface (z=0) during stance.

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
# To select frames where multiple conditions hold simultaneously, we can
# combine any number of boolean masks across limb coordinates using ``&``.

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
# pairs are simultaneously in swing. During trotting, quadrupeds typically
# move diagonal limb pairs together (e.g. front-left with hind-right), so
# we would expect the body to oscillate laterally (in ``y``) in opposite
# directions for each diagonal pair.

# Select the tail base position as a proxy for the rearward position of the
# body
tail = ds.position.sel(keypoints="Tail1")

# Create boolean masks for each limb in swing
fl_swing = tail["gait_FL"] == "swing"
fr_swing = tail["gait_FR"] == "swing"
hl_swing = tail["gait_HL"] == "swing"
hr_swing = tail["gait_HR"] == "swing"

# Select frames where each diagonal pair is simultaneously in swing
fl_hr = tail.sel(time=fl_swing & hr_swing)
fr_hl = tail.sel(time=fr_swing & hl_swing)

fig, ax = plt.subplots(figsize=(8, 3))

# Plot the tail base position in the x-y plane during diagonal swing phases
ax.scatter(fl_hr.time, fl_hr.sel(space="y"), s=8, label="FL and HR swing")
ax.scatter(fr_hl.time, fr_hl.sel(space="y"), s=8, label="FR and HL swing")

ax.set_xlabel("Time (s)")
ax.set_ylabel("Y position (mm)")
ax.legend()
plt.tight_layout()
plt.show()
# sphinx_gallery_thumbnail_number = 2

# %%
# As expected, the tail base occupies distinct y-positions for each diagonal
# pair, shifting towards the right during FL and HR swing and towards the
# left during FR and HL swing. This lateral oscillation is consistent with
# the expected weight transfer during trotting gait.

# %%
# Finally, :meth:`~xarray.DataArray.isin` can be used to select frames where
# a limb is in any one of multiple phases at once. For example, to retain
# only frames where the front-right limb phase was confidently identified
# (i.e. excluding ``"unknown"``):

fr_visible = ds.position.sel(time=ds.gait_FR.isin(["stance", "swing"]))
print(fr_visible)
