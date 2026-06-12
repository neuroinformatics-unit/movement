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

# Get coordinates from original input
time_coords = np.sort(loader.properties["time"].unique())
space_coords = ["x", "y"]
kpt_coords = loader.properties["keypoint"].unique().tolist()
indiv_coords = loader.properties["individual"].unique().tolist()

# Get number of frames
n_frames = time_coords.shape[0]
# This should match:
# n_frames = int(loader.properties["time"].max() * loader.fps) + 1

# Get properties dataframe
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

# add kpt and individuals (should match per row the properties ds)
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
# Compute confidence data array
# TODO: edited points should probably have NAN confidence

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

# Where output confidence is NaN, input confidence is NaN or 0
conf_nan_out = ds.confidence.isnull()
print(np.unique(ds_input.confidence.where(conf_nan_out).values))

# Where output confidence is not NaN, they are the same
np.testing.assert_equal(
    ds_input.confidence.where(~conf_nan_out).values,
    ds.confidence.where(~conf_nan_out).values,
)

# %%
# The mask that drops nans is from position
# A row is removed whenever x or y is NaN.
# In this SLEAP file there are frames where the position
# is missing but SLEAP still recorded a confidence/score
# (6,290 of them). Those whole rows never make it into
# the Points layer, so their confidence is gone — and your to_xarray().reindex() correctly fills those cells back as NaN.

# %%
# - In this SLEAP file there are frames where the position is missing but
#   SLEAP still recorded a confidence/score (sets it to 0?).
# - In the properties dataframe, we drop values that have POSITION = NAN
# - I think confidence should be set to nan

nan_position = ds.position.isnull().any(
    "space"
)  # position missing (x or y NaN)
nan_conf = ds.confidence.isnull()
print((nan_position & ~nan_conf).sum())  # missing position, present confidence

# %%
