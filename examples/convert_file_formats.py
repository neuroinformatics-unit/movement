"""Convert pose tracks between file formats
===========================================

Load pose tracks from one file format, modify them,
and save them to another file format.
"""

# %%
# Motivation
# ----------
# When working with pose estimation data, it's often useful to convert
# between file formats. For example, you may need to
# use some downstream analysis tool that requires a specific file
# format, or expects the keypoints to be named in a certain way.
#
# In the following example, we will load a dataset from a
# SLEAP file, modify the keypoints (rename, delete, reorder),
# and save the modified dataset as a DeepLabCut file.

# %%
# Imports
# -------
import pathlib

from movement import sample_data
from movement.io import load_poses, save_poses

# %%
# Load the dataset
# --------------------
# This should the location of a file output by one of
# our supported pose estimation
# supported pose estimation
# frameworks (e.g., DeepLabCut, SLEAP), containing predicted pose tracks.
# For example, the path could be something like:

# file_path = "/path/to/my/data.h5"

# %%
# For the sake of this example, we will use the path to one of
# the sample datasets provided with ``movement``.

fpath = sample_data.fetch_dataset_paths("SLEAP_single-mouse_EPM.analysis.h5")[
    "poses"
]
print(fpath)

# %%
# Now let's load this file into an xarray dataset, which we can then
# modify to our liking.
ds = load_poses.from_sleap_file(fpath, fps=30)
print(ds)
# %%
# Rename keypoints
# --------------------------------
# Create a dictionary that maps old keypoint names to new ones
rename_dict = {
    "snout": "nose",
    "left_ear": "earL",
    "right_ear": "earR",
    "centre": "middle",
    "tail_base": "tailbase",
    "tail_end": "tailend",
}


# %%
# Now we can run the following function,
# to rename the keypoints as defined in ``rename_dict``.


# the keypoints have been renamed.
# this function takes the dataset and the rename_dict as input.
def rename_keypoints(ds, rename_dict):
    # get the current names of the keypoints
    keypoint_names = ds.coords["keypoints"].values
    print("Original keypoints:", keypoint_names)
    # rename the keypoints
    if not rename_dict:
        print("No keypoints to rename. Skipping renaming step.")
    else:
        new_keypoints = [rename_dict.get(kp, str(kp)) for kp in keypoint_names]
        print("New keypoints:", new_keypoints)
        # Assign the modified values back to the Dataset
        ds = ds.assign_coords(keypoints=new_keypoints)
    return ds


# %%
# To prove to ourselves that the keypoints have been renamed,
# we can print the keypoints in the modified dataset.
ds_renamed = rename_keypoints(ds, rename_dict)
print("Keypoints in modified dataset:", ds_renamed.coords["keypoints"].values)


# %%
# Delete Keypoints
# -----------------
# Let's create a list of keypoints to delete.
# to delete modify this list accordingly

kps_to_delete = ["tailend"]


# %%
# Now we can go ahead and delete these keypoints
# using an appropriate function.
def delete_keypoints(ds, delete_keypoints):
    if not delete_keypoints:
        print("No keypoints to delete. Skipping deleting step.")
    else:
        # Delete the specified keypoints
        # and their corresponding data
        ds = ds.drop_sel(keypoints=delete_keypoints)
    return ds


# %%
# To prove to ourselves that the keypoints have been deleted,
# we can print the keypoints in the modified dataset.

ds_deleted = delete_keypoints(ds_renamed, kps_to_delete)
print("Keypoints in modified dataset:", ds_deleted.coords["keypoints"].values)


# %%
# Reorder keypoints
# ------------------
# Again create a list with the
# Let's list the keypoints in the desired order.

ordered_keypoints = ["nose", "earR", "earL", "middle", "tailbase"]


# %%
# Now we can go ahead and reorder
# those keypoints
def reorder_keypoints(ds, ordered_keypoints):
    # reorder the keypoints
    if not ordered_keypoints:
        print("No keypoints to reorder. Skipping reordering step.")
    else:
        # Reorder the keypoints in the Dataset
        ds = ds.reindex(keypoints=ordered_keypoints)
    return ds


ds_reordered = reorder_keypoints(ds_deleted, ordered_keypoints)
print(
    "Keypoints in modified dataset:", ds_reordered.coords["keypoints"].values
)

# %%
# # One function to rule them all
# # -----------------------------
# # Now that we know how to rename, delete, and reorder keypoints,
# # let's put it all together in a single function,
# # and see how we'd use this in a real-world scenario.
# #
# # The following function will convert all files in a folder
# # (that end with a specified suffix) to the desired format.
# # Each file will be loaded, modified, and saved to a new file.


data_dir = "/path/to/your/data/"
target_dir = "/path/to/your/target/data/"


def convert_all(data_dir, target_dir, suffix=".slp"):
    source_folder = pathlib.Path(data_dir)
    fpaths = list(source_folder.rglob(f"*{suffix}"))

    for fpath in fpaths:
        fpath = pathlib.Path(fpath)
        target_path = pathlib.Path(target_dir)

        # this determines the filename of your modified file
        # change it if you like to change the filename
        dest_path = target_path / f"{fpath.stem}_dlc.csv"

        if dest_path.exists():
            print(f"Skipping {fpath} as {dest_path} already exists.")
            return

        if fpath.exists():
            print(f"processing: {fpath}")
            # load the data
            ds = load_poses.from_sleap_file(fpath)
            ds_renamed = rename_keypoints(ds, rename_dict)
            ds_deleted = delete_keypoints(ds_renamed, kps_to_delete)
            ds_reordered = reorder_keypoints(ds_deleted, ordered_keypoints)
            # save poses to dlc file format
            save_poses.to_dlc_file(ds_reordered, dest_path)

        else:
            raise ValueError(
                f"File '{fpath}' does not exist. "
                f"Please check the file path and try again."
            )


# %%
