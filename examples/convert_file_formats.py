"""Load and modify the pose track files.
==========================================
this helps to format your pose tracks in a way that is
compatible with other
packages from other developers or even other analysis pipelines
from collaborators that might use slightly
different sets of keypoints.
Load and rename, delete or reorder
keypoints in pose track files (such as .h5, .slp, or .csv).
"""

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

fpath = sample_data.fetch_dataset_paths(
    "SLEAP_single-mouse_EPM.analysis.h5"
)["poses"]
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
    "neck_L": "earL",
    "neck_R": "earR",
    "neck": "neck",
    "back_L": "hipL",
    "back_R": "hipR",
    "abdomen_post": "tail",
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
# Delete Keypoints
# -----------------
# --------------------------------
# Let's create a list of keypoints to delete.
# to delete modify this list accordingly

kps_to_delete = ["abdomen_pre", "abdomen", "tailbase", "front_L", "front_R"]


# %%
# Now we can go ahead and delete these keypoints
# using an appropriate function.
def delete_keypoints(ds, delete_keypoints):
    if not delete_keypoints
        print("No keypoints to delete. Skipping deleting step.")
    else:
        # Delete the specified keypoints
        # and their corresponding data
        ds = ds.drop_sel(keypoints=delete_keypoints)
    return ds


# %%
# Reorder keypoints
# ------------------
# --------------------------------
# Again create a list with the
# Let's list the keypoints in the desired order.

ordered_keypoints = ["nose", "earL", "earR", "neck", "hipL", "hipR", "tail"]


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
print("Keypoints in modified dataset:", ds_reordered.coords["keypoints"].values)
# %%
# and how can I use this when I only have my filepath????
# -----------
# Great! Now we know how we can modify our movement datasets!
# let's put it all together and see how we could use this in a real world
# scenario.
# This function will convert all files in the folder to the desired format.
# The function will rename the keypoints, delete specified keypoints,
# and reorder them in the dataset.
# The function will then save the modified dataset to a new file.


FPATH = "/path/to/your/data/"


def convert_all(FPATH, ext="_inference.slp"):
    source_folder = pathlib.Path(FPATH)
    fpaths = list(source_folder.rglob(f"*{ext}"))

    for fpath in fpaths:
        fpath = pathlib.Path(fpath)

        # this determines the filename of your modified file
        # change it if you like to change the filename
        dest_path = fpath.parent / "tracking_2D_8KP.csv"

        if dest_path.exists():
            print(f"Skipping {fpath} as {dest_path} already exists.")
            return

        if fpath.exists():
            print(f"processing: {fpath}")
            # load the data
            ds = load_poses.from_sleap_file(fpath, fps=60)
            ds_renamed = rename_keypoints(ds, rename_dict)
            ds_deleted = delete_keypoints(ds_renamed, kps_to_delete)
            ds_reordered = reorder_keypoints(ds_deleted, ordered_keypoints)
            # save poses to dlc file format
            # here we are also splitting the multi animal file into 2
            # separate files for each animal
            save_poses.to_dlc_file(
                ds_reordered, dest_path, split_individuals=True
            )

        else:
            raise ValueError(
                f"File '{fpath}' does not exist. "
                f"Please check the file path and try again."
            )
