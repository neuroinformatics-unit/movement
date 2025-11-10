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
#
# We'll first walk through each step separately, and then
# combine them into a single function that can be applied
# to multiple files at once.

# %%
# Imports
# -------
import tempfile
from pathlib import Path

from movement import sample_data
from movement.io import load_poses, save_poses

# %%
# Load the dataset
# ----------------
# We'll start with the path to a file output by one of
# our :ref:`supported pose estimation frameworks<target-supported-formats>`.
# For example, the path could be something like:

# uncomment and edit the following line to point to your own local file
# file_path = "/path/to/my/data.h5"

# %%
# For the sake of this example, we will use the path to one of
# the sample datasets provided with ``movement``.

file_path = sample_data.fetch_dataset_paths(
    "SLEAP_single-mouse_EPM.analysis.h5"
)["poses"]
print(file_path)

# %%
# Now let's load this file into a
# :ref:`movement poses dataset<target-poses-and-bboxes-dataset>`,
# which we can then modify to our liking.

ds = load_poses.from_sleap_file(file_path, fps=30)
print(ds, "\n")
print("Individuals:", ds.coords["individuals"].values)
print("Keypoints:", ds.coords["keypoints"].values)


# %%
# .. note::
#     If you're running this code in a Jupyter notebook,
#     you can just type ``ds`` (instead of printing it)
#     to explore the dataset interactively.

# %%
# Rename keypoints
# ----------------
# We start with a dictionary that maps old keypoint names to new ones.
# Next, we define a function that takes that dictionary and a dataset
# as inputs, and returns a modified dataset. Notice that under the hood
# this function calls :meth:`xarray.Dataset.assign_coords`.

rename_dict = {
    "snout": "nose",
    "left_ear": "earL",
    "right_ear": "earR",
    "centre": "middle",
    "tail_base": "tailbase",
    "tail_end": "tailend",
}


def rename_keypoints(ds, rename_dict):
    # get the current names of the keypoints
    keypoint_names = ds.coords["keypoints"].values

    # rename the keypoints
    if not rename_dict:
        print("No keypoints to rename. Skipping renaming step.")
    else:
        new_keypoints = [rename_dict.get(kp, str(kp)) for kp in keypoint_names]
        # Assign the modified values back to the Dataset
        ds = ds.assign_coords(keypoints=new_keypoints)
    return ds


# %%
# Let's apply the function to our dataset and see the results.
ds_renamed = rename_keypoints(ds, rename_dict)
print("Keypoints in modified dataset:", ds_renamed.coords["keypoints"].values)


# %%
# Delete keypoints
# -----------------
# Let's create a list of keypoints to delete.
# In this case, we choose to get rid of the ``tailend`` keypoint,
# which is often hard to reliably track.
# We delete it using :meth:`xarray.Dataset.drop_sel`,
# wrapped in an appropriately named function.

keypoints_to_delete = ["tailend"]


def delete_keypoints(ds, delete_keypoints):
    if not delete_keypoints:
        print("No keypoints to delete. Skipping deleting step.")
    else:
        # Delete the specified keypoints and their corresponding data
        ds = ds.drop_sel(keypoints=delete_keypoints)
    return ds


ds_deleted = delete_keypoints(ds_renamed, keypoints_to_delete)
print("Keypoints in modified dataset:", ds_deleted.coords["keypoints"].values)


# %%
# Reorder keypoints
# ------------------
# We start with a list of keypoints in the desired order
# (in this case, we'll just swap the order of the left and right ears).
# We then use :meth:`xarray.Dataset.reindex`, wrapped in yet another function.

ordered_keypoints = ["nose", "earR", "earL", "middle", "tailbase"]


def reorder_keypoints(ds, ordered_keypoints):
    if not ordered_keypoints:
        print("No keypoints to reorder. Skipping reordering step.")
    else:
        ds = ds.reindex(keypoints=ordered_keypoints)
    return ds


ds_reordered = reorder_keypoints(ds_deleted, ordered_keypoints)
print(
    "Keypoints in modified dataset:", ds_reordered.coords["keypoints"].values
)

# %%
# Save the modified dataset
# ---------------------------
# Now that we have modified the dataset to our liking,
# let's save it to a .csv file in the DeepLabCut format.
# In this case, we save the file to a temporary
# directory, and we use the same file name
# as the original, but ending in ``_dlc.csv``.
# You will need to specify a different ``target_dir`` and edit
# the ``dest_path`` variable to your liking.

target_dir = tempfile.mkdtemp()
dest_path = Path(target_dir) / f"{file_path.stem}_dlc.csv"

save_poses.to_dlc_file(ds_reordered, dest_path, split_individuals=False)
print(f"Saved modified dataset to {dest_path}.")

# %%
# .. note::
#     The ``split_individuals`` argument allows you to save
#     a dataset with multiple individuals as separate files,
#     with the individual ID appended to each file name.
#     In this case, we set it to ``False`` because we only have
#     one individual in the dataset, and we don't need its name
#     appended to the file name.


# %%
# One function to rule them all
# -----------------------------
# Since we know how to rename, delete, and reorder keypoints,
# let's put it all together in a single function
# and see how we could apply it to multiple files at once,
# as we might do in a real-world scenario.
#
# The following function will convert all files in a folder
# (that end with a specified suffix) from SLEAP to DeepLabCut format.
# Each file will be loaded, modified according to the
# ``rename_dict``, ``keypoints_to_delete``, and ``ordered_keypoints``
# we've defined above, and saved to the target directory.


data_dir = "/path/to/your/data/"
target_dir = "/path/to/your/target/data/"


def convert_all(data_dir, target_dir, suffix=".slp"):
    source_folder = Path(data_dir)
    file_paths = list(source_folder.rglob(f"*{suffix}"))

    for file_path in file_paths:
        file_path = Path(file_path)

        # this determines the file names for the modified files
        dest_path = Path(target_dir) / f"{file_path.stem}_dlc.csv"

        if dest_path.exists():
            print(f"Skipping {file_path} as {dest_path} already exists.")
            continue

        if file_path.exists():
            print(f"Processing: {file_path}")
            # load the data from SLEAP file
            ds = load_poses.from_sleap_file(file_path)
            # modify the data
            ds_renamed = rename_keypoints(ds, rename_dict)
            ds_deleted = delete_keypoints(ds_renamed, keypoints_to_delete)
            ds_reordered = reorder_keypoints(ds_deleted, ordered_keypoints)
            # save modified data to a DeepLabCut file
            save_poses.to_dlc_file(
                ds_reordered, dest_path, split_individuals=False
            )
        else:
            raise ValueError(
                f"File '{file_path}' does not exist. "
                f"Please check the file path and try again."
            )
