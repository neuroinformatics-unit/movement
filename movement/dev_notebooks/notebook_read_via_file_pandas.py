# ruff: noqa
# %%
# imports
import ast
import re

import numpy as np
import pandas as pd

# %%
# data
# All frame numbers, target IDs and bounding boxes are 1-based
# (like multi-object tracking challenge https://motchallenge.net/instructions/)
# pixel coordinates are 0-based and pixel-centred?
file_path = "/Users/sofia/swc/project_movement_dataloader/05.09.2023-05-Left_track_corrected_NK.csv"
# "/Users/sofia/swc/project_movement_dataloader/04.09.2023-04-Right_RE_test_ORIGINAL-test_frame.csv"
# "/Users/sofia/swc/project_movement_dataloader/04.09.2023-04-Right_RE_test_corrected_ST_csv.csv"
# "/Users/sofia/swc/project_movement_dataloader/05.09.2023-05-Left_track_corrected_NK_editSM.csv"
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Read file as pandas data frame

df_file = pd.read_csv(file_path, sep=",", header=0)
df_file.columns


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Extract frame number
# improve this;

# frame number is between "_" and ".", led by at least one zero, followed by extension
pattern = r"_(0\d*\).\w+$"  # before: r"_(0\d*)\."

list_frame_numbers = []
for f in df_file.filename:
    if re.search(pattern, f):
        list_frame_numbers.append(int(re.search(pattern, f).group(1)))  # type: ignore

list_unique_frames = list(set(list_frame_numbers))


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Extract x,y,w,h of bboxes as numpy arrays
list_bbox_xy = []
list_bbox_wh = []
list_bbox_ID = []
for _, row in df_file.iterrows():
    # check shape is a rectangle
    assert ast.literal_eval(row.region_shape_attributes)["name"] == "rect"

    # extract bbox x,y coordinates
    list_bbox_xy.append(
        (
            ast.literal_eval(row.region_shape_attributes)["x"],
            ast.literal_eval(row.region_shape_attributes)["y"],
        )
    )

    # extract width and height
    list_bbox_wh.append(
        (
            ast.literal_eval(row.region_shape_attributes)["width"],
            ast.literal_eval(row.region_shape_attributes)["height"],
        )
    )

    # extract ID
    list_bbox_ID.append(int(ast.literal_eval(row.region_attributes)["track"]))

bbox_xy_array = np.array(list_bbox_xy)  # sort all by frame number
bbox_wh_array = np.array(list_bbox_wh)
bbox_ID_array = np.array(list_bbox_ID).reshape(-1, 1)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Make a dataframe with numpy arrays
df = pd.DataFrame(
    {
        "frame_number": list_frame_numbers,
        "x": bbox_xy_array[:, 0],
        "y": bbox_xy_array[:, 1],
        "w": bbox_wh_array[:, 0],
        "h": bbox_wh_array[:, 1],
        "ID": bbox_ID_array[:, 0],
    }
)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Build centroid and shape arrays
# (n_frames, n_unique_IDs, n_space)

list_unique_bbox_IDs = list(set(list_bbox_ID))
list_centroid_arrays = []
list_shape_arrays = []


assert 0 not in list_unique_bbox_IDs

for bbox_id in list_unique_bbox_IDs:
    # Get subset dataframe for one bbox (frane_number, x, y, w, h, ID)
    df_one_bbox = df.loc[df["ID"] == bbox_id]
    df_one_bbox = df_one_bbox.sort_values(by="frame_number")

    # Drop rows with same frame_number and ID
    # (if manual annotation, sometimes the same ID appears >1 in a frame)
    if len(df_one_bbox.frame_number.unique()) != len(df_one_bbox):
        print(f"ID {bbox_id} appears more than once in a frame")
        print("Dropping duplicates")
        df_one_bbox = df_one_bbox.drop_duplicates(
            subset=["frame_number", "ID"],  # they may differ in x,y,w,h
            keep="first",  # or last?
        )

    # Reindex based on full set of unique frames
    # (otherwise only the ones for this bbox are in the df)
    df_one_bbox_reindexed = (
        df_one_bbox.set_index("frame_number")
        .reindex(list_unique_frames)
        .reset_index()
    )

    # Convert to numpy arrays
    list_centroid_arrays.append(df_one_bbox_reindexed[["x", "y"]].to_numpy())
    list_shape_arrays.append(df_one_bbox_reindexed[["w", "h"]].to_numpy())

# Concatenate centroid arrays and shape arrays for all IDs
centroid_array = np.stack(list_centroid_arrays, axis=1)
shape_array = np.stack(list_shape_arrays, axis=1)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Refactoring ....
def _float_region_shape_attributes_to_numpy(
    df, list_region_shape_attributes: list[str]
) -> np.ndarray:
    list_bbox_attr = []
    for _, row in df.iterrows():
        # check shape is a rectangle
        assert ast.literal_eval(row.region_shape_attributes)["name"] == "rect"

        # extract bbox relevant attributes
        list_bbox_attr.append(
            tuple(
                float(ast.literal_eval(row.region_shape_attributes)[reg])
                for reg in list_region_shape_attributes
            )
        )

    bbox_attr_array = np.array(list_bbox_attr)

    return bbox_attr_array


def _int_region_attributes_to_numpy(
    df, region_attributes: list[str]
) -> np.ndarray:
    list_bbox_attr = []

    # extract bbox relevant region attributes
    for _, row in df.iterrows():
        list_bbox_attr.append(
            tuple(
                int(ast.literal_eval(row.region_attributes)[reg])
                for reg in region_attributes
            )
        )

    bbox_attr_array = np.array(list_bbox_attr)

    return bbox_attr_array


# %%%%%%%%%
# To recycle as tests
df = pd.read_csv(file_path, sep=",", header=0)

bbox_xy_arr = _float_region_shape_attributes_to_numpy(df, ["x", "y"])
print(np.all(np.equal(bbox_xy_arr, bbox_xy_array)))

bbox_wh_arr = _float_region_shape_attributes_to_numpy(df, ["width", "height"])
print(np.all(np.equal(bbox_wh_arr, bbox_wh_array)))

bbox_x_arr = _float_region_shape_attributes_to_numpy(df, ["x"])
print(np.all(np.equal(bbox_x_arr, bbox_xy_array[:, 0].reshape(-1, 1))))

bbox_ID_arr = _int_region_attributes_to_numpy(df, ["track"])
print(np.all(np.equal(bbox_ID_arr, bbox_ID_array)))
# %%
