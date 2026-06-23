"""Handle SLEAP analysis files with mixed labels
================================================
Explore a dataset that contains a mix of predicted and manually labelled
instances, examine NaN confidence values, and see what happens to those
points when confidence-based filtering is applied.
"""

# %%
# Imports
# -------
import numpy as np
from sleap_io.io.slp import read_labels

from movement import sample_data
from movement.filtering import filter_by_confidence

# %%
# Load the sample dataset
# -----------------------
# This dataset was exported from a SLEAP project that contains both
# predicted pose tracks and manually labelled (proofread) instances.
# In SLEAP, manually labelled instances have no model confidence score,
# so movement stores them as ``NaN`` in the ``confidence`` data variable.

ds = sample_data.fetch_dataset(
    "SLEAP_three-mice_Aeon_mixed-labels.analysis.h5"
)
ds

# %%
# The dataset tracks three mice (``individual`` dimension) at a single
# keypoint (``centroid``) over ~12 seconds of video at 50 fps.

# %%
# Count NaNs in the confidence array
# -----------------------------------
# Let's first get a feel for how many NaN values appear in ``confidence``
# and whether they are spread evenly across individuals.

conf = ds.confidence
print("=== NaN values in confidence ===")
total = conf.size
total_nan = int(conf.isnull().sum())
print(f"Overall: {total_nan} / {total} ({100 * total_nan / total:.1f} %)")
print()
for ind in ds.individual.values:
    c = conf.sel(individual=ind)
    n = int(c.isnull().sum())
    print(f"  {ind}: {n} / {c.size} ({100 * n / c.size:.1f} %)")

# %%
# Overall ~14 % of confidence values are NaN, unevenly spread across
# individuals.  Why so many?

# %%
# Why are some confidence values NaN?
# ------------------------------------
# When movement loads a SLEAP file that contains manually labelled instances,
# those instances are stored with ``NaN`` confidence (the model never scored
# them).  Predicted instances carry the actual model score.
#
# The presence of NaN confidence therefore tells us *which frames were
# manually annotated*, not that the tracking model was uncertain.

# %%
# Distinguish "missing data" NaNs from "manually labelled" NaNs
# --------------------------------------------------------------
# A NaN confidence value can mean two different things:
#
# * The frame had no detection at all (no position data either).
# * The frame was manually labelled: a valid position exists but there is
#   no model confidence score.
#
# Let's separate these two cases for each individual.

pos_x = ds.position.sel(space="x")  # (time, keypoint, individual)
nan_conf = conf.isnull()
nan_pos = pos_x.isnull()

print("=== NaN confidence breakdown per individual ===")
for ind in ds.individual.values:
    nc = nan_conf.sel(individual=ind)
    np_ = nan_pos.sel(individual=ind)
    both = int((nc & np_).sum())
    conf_nan_pos_ok = int((nc & ~np_).sum())
    print(f"\n{ind}:")
    print(f"  NaN conf + NaN pos   (no detection):       {both}")
    print(f"  NaN conf + valid pos (manually labelled):  {conf_nan_pos_ok}")

# %%
# For ``AEON3B_TP2``, 11 out of 43 NaN-confidence frames still carry a valid
# position -- those are the manually labelled instances that were added or
# corrected by a user.

# %%
# Inspect some manually labelled frames
# ---------------------------------------
# Let's look at a short window around one of those NaN-confidence frames
# with a valid position to see the context.

ind = "AEON3B_TP2"
c = conf.sel(individual=ind, keypoint="centroid")
p = ds.position.sel(individual=ind, keypoint="centroid", space="x")

manually_labelled_mask = c.isnull() & ~p.isnull()
first_t = float(ds.time.values[manually_labelled_mask.values][0])
first_idx = list(ds.time.values).index(first_t)
window = range(max(0, first_idx - 2), min(len(ds.time), first_idx + 5))

print(f"Context around first manually labelled frame for {ind}:")
print(f"{'time':>8}  {'confidence':>12}  {'pos_x':>8}")
for i in window:
    t = float(ds.time.values[i])
    cv = float(c.isel(time=i))
    pv = float(p.isel(time=i))
    marker = (
        "  <-- NaN conf, valid pos"
        if np.isnan(cv) and not np.isnan(pv)
        else ""
    )
    conf_str = "nan" if np.isnan(cv) else f"{cv:.4f}"
    print(f"{t:8.2f}  {conf_str:>12}  {pv:8.1f}{marker}")

# %%
# The manually labelled frame sits between two normally predicted frames.
# Its position is perfectly sensible, but its confidence is ``NaN``.

# %%
# What happens when we filter by confidence?
# -------------------------------------------
# :func:`movement.filtering.filter_by_confidence` keeps only data points
# where ``confidence >= threshold``.  Because ``NaN >= 0.6`` evaluates to
# ``False`` in xarray, *any point with NaN confidence is silently dropped*,
# even if it has a perfectly valid position.

pos = ds.position
pos_filtered = filter_by_confidence(pos, conf, threshold=0.6)

new_nans = pos_filtered.isnull() & ~pos.isnull()
nan_conf_broadcast = nan_conf.broadcast_like(pos)

new_nans_from_nan_conf = new_nans & nan_conf_broadcast
new_nans_from_low_conf = new_nans & ~nan_conf_broadcast

total_new = int(new_nans.sum())
from_nan_conf = int(new_nans_from_nan_conf.sum())
from_low_conf = int(new_nans_from_low_conf.sum())

print("=== New NaN positions introduced by filter_by_confidence ===")
print(f"  Total new NaN position values: {total_new}")
print(f"    Due to NaN confidence (manually labelled):  {from_nan_conf}")
print(f"    Due to low confidence (< 0.6, predicted):   {from_low_conf}")

# %%
# Out of {total_new} new NaN positions, {from_nan_conf} came from manually
# labelled frames that had a valid position -- they were destroyed simply
# because their confidence was NaN.  Only {from_low_conf} were genuinely
# low-confidence predictions.
#
# This is the core problem raised in the design discussion for the
# proofreading widget: **how should confidence be stored for manually
# edited or added points so that they survive confidence-based filtering?**

# %%
# Three candidate strategies
# ---------------------------
# Below we illustrate the consequences of three options that have been
# proposed in the discussion.

# %%
# **Option A -- set confidence to 1.0 for manually labelled points**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The simplest fix: assign a fixed sentinel value of ``1.0``.  After
# filtering at any threshold <= 1.0, manually labelled points are preserved.
# A small caveat: SLEAP confidence scores can exceed 1.0, so ``1.0`` is not
# a universal "maximum" -- but it is above any typical filtering threshold.

conf_option_a = conf.copy()
manually_labelled = nan_conf & ~nan_pos  # (time, keypoint, individual)
conf_option_a = conf_option_a.where(~manually_labelled, other=1.0)

pos_a = filter_by_confidence(pos, conf_option_a, threshold=0.6)
new_nans_a = int((pos_a.isnull() & ~pos.isnull()).sum())
print(f"Option A (conf=1.0): {new_nans_a} new NaN positions after filtering")

# %%
# No manually labelled positions are lost.

# %%
# Confirm findings against the source .slp file
# -----------------------------------------------
# The analysis ``.h5`` file was exported from a SLEAP labels (``.slp``) file
# that stores the original data structures.  Let's load that file directly
# with ``sleap_io`` and confirm that the NaN confidence values really do
# correspond to manually labelled (``Instance``) objects, whereas predicted
# frames use ``PredictedInstance`` objects that carry an explicit score.

slp_path = sample_data.fetch_dataset_paths(
    "SLEAP_three-mice_Aeon_mixed-labels.predictions.slp"
)["poses"]
labels = read_labels(str(slp_path))

total_user = sum(len(lf.user_instances) for lf in labels.labeled_frames)
total_predicted = sum(
    len(lf.predicted_instances) for lf in labels.labeled_frames
)
print(f"Total USER (Instance) objects:           {total_user}")
print(f"Total PREDICTED (PredictedInstance) objects: {total_predicted}")

# %%
# Inspect the data structures of one user and one predicted instance
# to understand why confidence is absent for the former.

lf29 = next(lf for lf in labels.labeled_frames if lf.frame_idx == 29)

pred = lf29.predicted_instances[0]
user = lf29.user_instances[0]
pred_pt = pred.points[0]
user_pt = user.points[0]

print("PREDICTED instance (PredictedInstance):")
print(f"  Python type:          {type(pred).__name__}")
print(f"  Point dtype fields:   {pred_pt.dtype.names}")
print(f"  Point score:          {float(pred_pt['score']):.4f}")
print(f"  Instance-level score: {pred.score:.4f}")
print()
print("USER instance (Instance):")
print(f"  Python type:        {type(user).__name__}")
print(f"  Point dtype fields: {user_pt.dtype.names}")
print(f"  Has .score attr:    {hasattr(user, 'score')}")
print(f"  'score' in point:   {'score' in user_pt.dtype.names}")

# %%
# ``PredictedInstance`` point records contain a ``'score'`` field;
# ``Instance`` point records only store ``('xy', 'visible', 'complete',
# 'name')``.  The score field is structurally absent, not merely unknown.
# This is why movement's loader fills those entries with ``np.nan``.

# %%
# Cross-reference: .slp user instances vs .h5 NaN-confidence rows
# ----------------------------------------------------------------
# If the mapping is correct, every user instance in the ``.slp`` file
# should correspond exactly to a (frame, individual) pair in the ``.h5``
# where confidence is NaN and position is valid -- and vice versa.

# Collect (frame_idx, track, x, y) from .slp user instances
slp_user_rows = []
for lf in labels.labeled_frames:
    for inst in lf.user_instances:
        xy = inst.points[0]["xy"]
        slp_user_rows.append(
            (lf.frame_idx, inst.track.name, float(xy[0]), float(xy[1]))
        )

# Collect (frame_idx, individual, x, y) from .h5 NaN-conf valid-pos entries
h5_manual_rows = []
for ind in ds.individual.values:
    mask = manually_labelled.sel(individual=ind, keypoint="centroid")
    for i, flag in enumerate(mask.values):
        if flag:
            t = float(ds.time.values[i])
            frame_idx = round(t * float(ds.fps))
            x = float(
                ds.position.sel(
                    individual=ind, keypoint="centroid", space="x"
                ).isel(time=i)
            )
            y = float(
                ds.position.sel(
                    individual=ind, keypoint="centroid", space="y"
                ).isel(time=i)
            )
            h5_manual_rows.append((frame_idx, ind, x, y))

slp_set = set((r[0], r[1]) for r in slp_user_rows)
h5_set = set((r[0], r[1]) for r in h5_manual_rows)

print(f"User instances in .slp:          {len(slp_user_rows)}")
print(f"NaN-conf valid-pos rows in .h5:  {len(h5_manual_rows)}")
print(f"(frame, individual) pairs match: {slp_set == h5_set}")
print()
print(
    f"{'frame':>6}  {'individual':>16}  {'x (.slp)':>10}  "
    f"{'x (.h5)':>10}  {'match':>6}"
)
for (fi, tr, sx, sy), (_, __, hx, hy) in zip(
    sorted(slp_user_rows), sorted(h5_manual_rows), strict=True
):
    match = abs(sx - hx) < 0.05 and abs(sy - hy) < 0.05
    print(
        f"{fi:6d}  {tr:>16}  {sx:10.1f}  {hx:10.1f}  "
        f"{'yes' if match else 'NO':>6}"
    )

# %%
# Every row in the ``.slp`` user-instance table maps exactly to a NaN-conf
# entry in the ``.h5`` dataset, and the coordinates agree to sub-pixel
# precision.  The source of the NaN confidence values is confirmed:
# they are the ``Instance`` objects (manually labelled frames) whose
# ``sleap_io`` point records have no ``'score'`` field.

# %%
# **Option B -- keep confidence NaN, but guard the filter**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Keep confidence as NaN for manually labelled points, but update
# :func:`~movement.filtering.filter_by_confidence` to exclude NaN-confidence
# points from filtering (i.e. treat them as "always keep").

conf_is_nan = nan_conf.broadcast_like(pos)
pos_b = pos.where((conf >= 0.6) | conf_is_nan)
new_nans_b = int((pos_b.isnull() & ~pos.isnull()).sum())
print(
    f"Option B (guard filter): {new_nans_b} new NaN positions after filtering"
)

# %%
# Same result as Option A, but requires a code change to
# :func:`~movement.filtering.filter_by_confidence`.

# %%
# **Option C -- add a boolean ``manually_labelled`` data variable**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Store origin information explicitly as a separate boolean array.
# Filtering logic can then skip flagged points.

ds_extended = ds.copy()
ds_extended["manually_labelled"] = manually_labelled

# Guard the filter:
# keep points where confidence >= threshold OR manually labelled
origin_broadcast = ds_extended["manually_labelled"].broadcast_like(pos)
pos_c = pos.where((conf >= 0.6) | origin_broadcast)
new_nans_c = int((pos_c.isnull() & ~pos.isnull()).sum())
print(
    f"Option C (origin flag): {new_nans_c} new NaN positions after filtering"
)
print()
print("Manually labelled flag stored in the dataset:")
print(ds_extended["manually_labelled"])

# %%
# Also preserves all manually labelled positions, and makes provenance
# explicit and queryable.  The trade-off is a slightly richer dataset
# schema and the need to propagate the flag through downstream operations.

# %%
# Summary
# -------
# This exploration confirms that, in a real SLEAP file with mixed labels,
# NaN confidence reliably flags manually labelled instances.  Calling
# :func:`~movement.filtering.filter_by_confidence` with the current
# implementation silently discards those points.
#
# * **Option A (confidence = 1.0)** is the most backwards-compatible:
#   no changes to movement's data schema or filtering logic are needed.
#   It conflates "manually labelled" with "very high confidence", which
#   may be misleading for downstream analyses that interpret confidence
#   values literally.
#
# * **Option B (NaN confidence + guard in filter)** preserves the semantic
#   distinction (NaN = no score) but requires a targeted change to
#   :func:`~movement.filtering.filter_by_confidence`.
#
# * **Option C (separate origin flag)** is the most expressive and
#   future-proof option.  It clearly separates "where did this point come
#   from?" from "how confident is the model?", and opens the door to richer
#   provenance tracking (e.g. distinguishing ``"predicted"``,
#   ``"manually_edited"``, ``"interpolated"``).  It requires schema changes
#   and downstream updates.
