# ruff: noqa
# %%
# imports
import ast
import re
from typing import Callable

import numpy as np
import pandas as pd

from movement.io.load_bboxes import (
    _extract_frame_number_from_via_tracks_df,
    _via_attribute_column_to_numpy,
)

# %%
# data
# All frame numbers, target IDs and bounding boxes are 1-based
# (like multi-object tracking challenge https://motchallenge.net/instructions/)
# pixel coordinates are 0-based and pixel-centred?
# file_path = "/Users/sofia/swc/project_movement_dataloader/05.09.2023-05-Left_track_corrected_NK.csv"
file_path = "/Users/sofia/swc/project_movement_dataloader/04.09.2023-04-Right_RE_test_ORIGINAL-test_frame.csv"
# "/Users/sofia/swc/project_movement_dataloader/04.09.2023-04-Right_RE_test_corrected_ST_csv.csv"
# "/Users/sofia/swc/project_movement_dataloader/05.09.2023-05-Left_track_corrected_NK_editSM.csv"


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Extract 2D arrays from df_file
df_file = pd.read_csv(file_path, sep=",", header=0)

# frame number array
frame_array = _extract_frame_number_from_via_tracks_df(df_file)  # 2D

# position 2D array
# rows: frames
# columns: x,y
bbox_position_array = _via_attribute_column_to_numpy(
    df_file, "region_shape_attributes", ["x", "y"], float
)
print(bbox_position_array.shape)

# shape 2D array
bbox_shape_array = _via_attribute_column_to_numpy(
    df_file, "region_shape_attributes", ["width", "height"], float
)
print(bbox_shape_array.shape)

# track 2D array
bbox_ID_array = _via_attribute_column_to_numpy(
    df_file, "region_attributes", ["track"], int
)
print(bbox_ID_array.shape)

# confidence 2D array
region_attributes_dicts = [
    ast.literal_eval(d) for d in df_file.region_attributes
]
if all(["confidence" in d for d in region_attributes_dicts]):
    bbox_confidence_array = _via_attribute_column_to_numpy(
        df_file, "region_attributes", ["confidence"], float
    )
else:
    bbox_confidence_array = np.full(frame_array.shape, np.nan)
print(bbox_confidence_array.shape)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Make a 2D dataframe
df = pd.DataFrame(
    {
        "frame_number": frame_array[:, 0],
        "x": bbox_position_array[:, 0],
        "y": bbox_position_array[:, 1],
        "w": bbox_shape_array[:, 0],
        "h": bbox_shape_array[:, 1],
        "confidence": bbox_confidence_array[:, 0],
        "ID": bbox_ID_array[:, 0],
    }
)

# important!
# sort by ID and frame number!!!!
df = df.sort_values(by=["ID", "frame_number"]).reset_index(drop=True)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Fill in with nans
# every ID should have all frames

multi_index = pd.MultiIndex.from_product(
    [df.ID.unique(), df.frame_number.unique()],
    names=["ID", "frame_number"],
)

df = df.set_index(["ID", "frame_number"]).reindex(multi_index).reset_index()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Build position, shape and confidence arrays
# (n_frames, n_unique_IDs, n_space)

# list_sorted_unique_IDs = df["ID"].unique().tolist()
# list_centroid_arrays = []
# list_shape_arrays = []


# for bbox_id in list_sorted_unique_IDs:
#     # Get subset dataframe for one bbox (frmne_number, x, y, w, h, ID)
#     df_one_bbox = df.loc[df["ID"] == bbox_id]

#     # Append 2D arrays to lists
#     list_centroid_arrays.append(df_one_bbox[["x", "y"]].to_numpy())
#     list_shape_arrays.append(df_one_bbox[["w", "h"]].to_numpy())

# # Concatenate centroid arrays and shape arrays for all IDs
# centroid_array = np.stack(list_centroid_arrays, axis=1)
# shape_array = np.stack(list_shape_arrays, axis=1)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Alternative - probably faster
# https://stackoverflow.com/questions/66105288/creating-grouped-stacked-arrays-from-pandas-data-frame

# compute indices where ID switches
bool_ID_diff_from_prev = df["ID"].ne(df["ID"].shift())  # pandas series
idcs_ID_switch = bool_ID_diff_from_prev.loc[lambda x: x].index[1:].to_numpy()

split_arrays = {}
for array_str, cols in zip(
    ["centroid", "shape", "confidence"],
    [["x", "y"], ["w", "h"], ["confidence"]],
):
    split_arrays[array_str] = np.split(
        df[cols].to_numpy(),
        idcs_ID_switch,  # along axis=0
    )

# stack along first axis
centroid_array = np.stack(split_arrays["centroid"])
shape_array = np.stack(split_arrays["shape"])
confidence_array = np.stack(split_arrays["confidence"])

# %%
list_split_centroid_arrays = np.split(
    df[["x", "y"]].to_numpy(),
    idcs_ID_switch,  # along axis=0
)

list_split_shape_arrays = np.split(
    df[["w", "h"]].to_numpy(),
    idcs_ID_switch,  # along axis=0
)

list_split_confidence_arrays = np.split(
    df[["confidence"]].to_numpy(),
    idcs_ID_switch,  # along axis=0
)


# not all IDs are present for all frames
# how do I fill in with nans?
# probs should do before this....
centroid_array = np.array(list_split_centroid_arrays)

# %%
