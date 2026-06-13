"""Exploring how to export edited keypoints as movement dataset.

Keypoints are only dragged, not added or deleted.

"""
# %%
import napari
import numpy as np
import pandas as pd
import xarray as xr

from movement import sample_data
from movement.io import load_dataset
from movement.napari.loader_widgets import DataLoader
from movement.utils.reports import report_nan_values

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Get path to sample file
# We select one with NaNs
file_path = sample_data.fetch_dataset_paths(
    "SLEAP_single-mouse_EPM.analysis.h5"
)["poses"]
print(file_path)

input_fps = 30.0


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# To confirm it actually has NaNs:
ds_input = load_dataset(file_path, fps=input_fps)
print(report_nan_values(ds_input.position))

# Missing points (marked as NaN) in position:
# keypoint                  snout           left_ear          right_ear             centre          tail_base            tail_end
# individual
# id_0        4494/18485 (24.31%)  513/18485 (2.78%)  533/18485 (2.88%)  490/18485 (2.65%)  704/18485 (3.81%)  2496/18485 (13.5%)

# %%
# NOTE: In the input the values where position is NaN
# have confidence 0; seems like SLEAP does this with
# confidence points below a threshold?
nan_position = ds_input.position.isnull().any("space")
print(np.nanmax(ds_input.confidence.where(nan_position).values))


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Instantiate napari viewer and loader widget
viewer = napari.Viewer()
loader = DataLoader(viewer)

# Fill in the loader widget as a user would
loader.file_path_edit.setText(str(file_path))
loader.source_software_combo.setCurrentText("SLEAP")
loader.fps_spinbox.setValue(input_fps)  # optional, defaults to 1.0

# Simulate clicking "Load"
loader._on_load_clicked()

napari.run()


# %%%%%%%%%%%%%%%%%%%%%%%%%%
# Note:
# - viewer.layers[0] and loader.points_layer point to the same object;
#   When a user edits points in the UI, those changes are reflected in
#   loader.points_layer.data
# - loader.points_layer.data, loader.points_layer.properties is **live** data, but
#   loader.data, loader.properties, .data_not_nan is the **original** passed data
#
# We inspect this in the following cells
# %%%%%%%%%%%
# Original input data (frozen)
print(loader.data.shape)
# ===> original data, includes nans, stays frozen with edits

print(loader.properties.shape)
# ====> properties for original data
# %%%%%%%%%
# Live data
print(loader.points_layer.data.shape)
# ===> live data, no nans, updates with edits
# same as viewer.layers[0].data

print(loader.points_layer.properties)
# ===> also live? (check), no nans, matches shape of loader.points_layer.data,
# it is a dict!
# same as viewer.layers[0].properties
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# We edit in the UI a keypoint
# "centre" keypoint at frame 108
# id_0, centre, frame=108 (time=3.6)

# What index is this in viewer.layers[0].properties?
df = pd.DataFrame.from_dict(viewer.layers[0].properties)
index_label = df.loc[df["keypoint"] == "centre", "time"].idxmin()

# get positional index
row_idx = df.index.get_loc(index_label)

# %%
# Print coordinates before edit
print(loader.points_layer.data[row_idx, :])
# ===> [108.         539.72717285 120.66210938]

# %%
# We edit in napari dragging the "centre" point upwards
# (y-coord should decrease after edit)
# %%
# Print coordinates after edit
print(loader.points_layer.data[row_idx, :])
# ===> [108.         258.61696477 119.80244513]

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# How to format the edited point data as a dataset?

# reconstructed_ds = napari_layers_to_ds(
#     points_data_not_nan, ===> loader.points_layer.data
#     points_properties_not_nan, ====> loader.points_layer.properties
#     valid_frames, ---?
# )

# Get coordinates from **original** input
time_coords = np.sort(loader.properties["time"].unique())
space_coords = ["x", "y"]
kpt_coords = loader.properties["keypoint"].unique().tolist()
indiv_coords = loader.properties["individual"].unique().tolist()

# Get number of frames
n_frames = time_coords.shape[0]
# This should match:
# n_frames = int(loader.properties["time"].max() * loader.fps) + 1

# Get **live** properties dataframe
properties_df = pd.DataFrame.from_dict(loader.points_layer.properties)


# %%%%%%%%%%%%%%%%%%%%%%%%%
# Compute position data array,
# from points layer data and properties

# NOTE: Reindex is exact label matching!!
# time coords need to match exactly
position_df = pd.DataFrame(
    loader.points_layer.data,
    columns=["frame", "y", "x"],
)
# add time
position_df["time"] = position_df["frame"] / loader.fps

# add kpt and individuals 
# (should match per row the *live* properties df)
# pandas doesn't assign positionally — it aligns by index label. 
position_df["keypoint"] = properties_df["keypoint"]
position_df["individual"] = properties_df["individual"]

# create a space column by melting x and y columns
position_df = position_df.melt(
    id_vars=["time", "frame", "keypoint", "individual"],
    var_name="space",
    value_name="position",
)

# create data array and fill missing values with reindexing of nans
position_da = (
    position_df.set_index(["time", "space", "keypoint", "individual"])[
        "position"
    ]
    .astype(np.float32)  # only to position
    .to_xarray()
    .reindex(
        time=time_coords,  # .astype(int),
        space=space_coords,
        keypoint=kpt_coords,
        individual=indiv_coords,
    )
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%
# Identify edited points to set their confidence to NaN

# Build dataframe with original data and properties as passed to 
# the points layer
original_df = pd.DataFrame(
    loader.data[loader.data_not_nan, 1:],  # drop track_id col
    columns=["frame", "y", "x"],
)
original_props = loader.properties.iloc[loader.data_not_nan].reset_index(
    drop=True  # we need to reset index to match original_df
)
# OJO: here pandas aligns by index label!
original_df["keypoint"] = original_props["keypoint"]
original_df["individual"] = original_props["individual"]


# Build dataframe with "Live" data from the points layer
live_df = pd.DataFrame(
    loader.points_layer.data,
    columns=["frame", "y", "x"],
)
# OJO: here pandas aligns by index label!
live_df["keypoint"] = properties_df["keypoint"]
live_df["individual"] = properties_df["individual"]

# Pair up rows; left merge keeps live_df rows in their original order
key_cols = ["frame", "individual", "keypoint"]
merged = live_df.merge(
    original_df,
    on=key_cols,
    how="left",
    suffixes=("_live", "_orig"),
)

# We assume tuples of values (time, indiv, kpt) are unique
assert len(merged) == len(properties_df)  

# %%
# Get masks for edited and added points
# Edited: present in both _live and _orig, but coordinates moved
# Added: no matching original row (y_orig/x_orig are NaN)
added_mask = merged["y_orig"].isna().to_numpy()
moved_mask = ~np.all(
    np.isclose(
        merged[["y_live", "x_live"]].to_numpy(),
        merged[["y_orig", "x_orig"]].to_numpy(),
    ),
    axis=1,
)
edited_mask = added_mask | moved_mask
print(f"{added_mask.sum()} added, {(moved_mask & ~added_mask).sum()} moved")

# %%
# Set confidence to NaN for edited/added points
properties_df = properties_df.copy()
properties_df.loc[edited_mask, "confidence"] = np.nan

# when a point is added, napari copies the properties
# of the last selected point and assigned them to it
# For future proofing, let's derive the time coord from
# the frames like for the position data
properties_df['time'] = position_df["frame"] / loader.fps

# %%%%%%%%%%%%%%%%%%%%%%%%%%%
# Compute confidence data array
# 
# Notes:
# - .to_xarray(): pivots the three index levels into dims and
#  places each confidence value at its matching cell
#  (missing combinations become NaN).
#
# - reindex(...) then pads to your complete coordinate set,
#  so any time/keypoint/individual not present in df_live stays NaN.

confidence_da = (
    properties_df.set_index(["time", "keypoint", "individual"])["confidence"]
    .to_xarray()
    .reindex(
        time=time_coords,
        keypoint=kpt_coords,
        individual=indiv_coords,
    )
)


# %%%%%%%%%%%%%%%%%%%%%%%%%%
# Build dataset
ds = xr.Dataset(
    dict(position=position_da, confidence=confidence_da),
    attrs={
        "source_software": loader.source_software,
        "ds_type": "poses",
        "fps": loader.fps,
        "time_unit": "seconds" if loader.fps != 1 else "frames",
        "source_file": loader.file_path,
    },
)

ds
# %%%%%%%%%%%%%%%%%%%%
# Check position is equal if no edits

xr.testing.assert_equal(ds.position, ds_input.position)


# %%%%%%%%%%%%%%%%%%
# Check confidence

# Where output confidence is NaN, input confidence is NaN or 0, or
# the value from the edited point
conf_nan_out = ds.confidence.isnull()
print(np.unique(ds_input.confidence.where(conf_nan_out).values))

# Where output confidence is not NaN, they are the same
np.testing.assert_equal(
    ds_input.confidence.where(~conf_nan_out).values,
    ds.confidence.where(~conf_nan_out).values,
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Claude

# The explicit value_vars=["x", "y"] matters now: points_df has a confidence column, and without it melt would treat confidence as a third "space" value.
# Order of operations with the edit-detection cell: keep the edited_mask cell before this one, operating on properties_df as it does now. The confidence values (including the NaNs you set for edited/added points) are copied into points_df row-by-row before the dedup, so everything stays aligned. The dedup must not run before edited_mask is computed, since that mask is positional against the un-deduplicated rows.
# keep="last" relies on napari appending added points after the original ones, which holds for Points.add. If a duplicate ever arose some other way (it shouldn't), "last" would be arbitrary — that's why the print of dropped rows is worth keeping, so a nonzero count is visible when you didn't add anything.

# %%%%%%%%%%%%%%%%%%%%%%%%%
# Build a point-level dataframe: one row per live point,
# with identity keys, coordinates and confidence together

points_df = pd.DataFrame(
    loader.points_layer.data,
    columns=["frame", "y", "x"],
)
points_df["time"] = points_df["frame"] / loader.fps

# properties rows are kept in sync with data rows by napari
points_df["keypoint"] = properties_df["keypoint"].to_numpy()
points_df["individual"] = properties_df["individual"].to_numpy()
points_df["confidence"] = properties_df["confidence"].to_numpy()

# A point added on a frame where that (keypoint, individual) already
# has a point creates a duplicate key, which to_xarray() can't handle.
# So we need to remove duplicates here.
# napari appends new points at the end, so keep="last" makes the
# user-added point override the original.
n_before = len(points_df)
points_df = points_df.drop_duplicates(
    subset=["time", "keypoint", "individual"],
    keep="last",
)
print(f"dropped {n_before - len(points_df)} overridden point(s)")

# %%%%%%%%%%%%%%%%%%%%%%%%%
# Position data array
position_df = points_df.melt(
    id_vars=["time", "frame", "keypoint", "individual"],
    value_vars=["x", "y"],
    var_name="space",
    value_name="position",
)
position_da = (
    position_df.set_index(["time", "space", "keypoint", "individual"])[
        "position"
    ]
    .astype(np.float32)
    .to_xarray()
    .reindex(
        time=time_coords,
        space=space_coords,
        keypoint=kpt_coords,
        individual=indiv_coords,
    )
)

# %%%%%%%%%%%%%%%%%%%%%%%%%
# Confidence data array — built from the same deduplicated keys,
# with time derived from the frame column (the `time` property of
# an added point is a stale copy of napari's current_properties)
confidence_da = (
    points_df.set_index(["time", "keypoint", "individual"])["confidence"]
    .to_xarray()
    .reindex(
        time=time_coords,
        keypoint=kpt_coords,
        individual=indiv_coords,
    )
)
