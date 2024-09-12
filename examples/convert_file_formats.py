"""Load and modify the pose track files.
==========================================
this helps to format your pose tracks in a way that is compatible with other
packages from other developers.
Load and rename, delete and reorder keypoints in pose track files.
"""

# %%
# Imports
# -------
import pathlib

from movement import sample_data
from movement.io import load_poses, save_poses

# %%
# Define the file path
# --------------------
# This should the location to your outputfile by one of our
# supported pose estimation
# frameworks (e.g., DeepLabCut, SLEAP), containing predicted pose tracks.
# For example, the path could be something like:

# uncomment and edit the following line to point to your own local file
# file_path = "/path/to/my/data.h5"

# %%
# For the sake of this example, we will use the path to one of
# the sample datasets provided with ``movement``.

fpath = sample_data.fetch_dataset_paths(
    "SLEAP_three-mice_Aeon_proofread.analysis.h5"
)["poses"]
print(fpath)
# %%
# this is the main function. it takes the file path, loads the data,
# allows you to change the name of keypoints
# and deletes some keypoints that are not needed.
# you can then also change the order of the kps if you need to


# %%
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
# Create a list of keypoints to delete modify accordingly
delete_keypoints = ["abdomen_pre", "abdomen", "tailbase", "front_L", "front_R"]
# %%
# Define the desired order of keypoints
ordered_keypoints = ["nose", "earL", "earR", "neck", "hipL", "hipR", "tail"]

# %%
# define the function to convert the inference file
# this function takes the file path, the rename_dict,
# delete_keypoints and ordered_keypoints
# as arguments and modifies the file accordingly


def convert_inference_file(
    fpath, rename_dict, delete_keypoints, ordered_keypoints
):
    fpath = pathlib.Path(fpath)

    # this determines the filename of your modified file
    # change it if you like to change the filename
    dest_path = fpath.parent / f"{fpath.stem}_modified.csv"

    # if the fpath you intend to change has already been changed
    # and a file with the same name already exists,
    # it will simply skip that file and exit the function
    if dest_path.exists():
        print(f"Skipping {fpath} as {dest_path} already exists.")
        return

    if fpath.exists():
        print(f"converting: {fpath}")
        # load the data
        ds = load_poses.from_sleap_file(fpath, fps=60)

        # get the names of the keypoints
        keypoint_names = ds.coords["keypoints"].values
        print("Original keypoints:", keypoint_names)

        # delete the keypoints that are not needed
        if delete_keypoints is None or len(delete_keypoints) == 0:
            print("No KPs to delete. Skipping deleting step.")
        else:
            # Delete the specified keypoints and their corresponding data
            ds = ds.drop_sel(keypoints=delete_keypoints)

        # rename the keypoints
        if rename_dict is None or len(rename_dict) == 0:
            print("No KPs to rename. Skipping renaming step.")
        else:
            new_keypoints = [rename_dict.get(kp, kp) for kp in keypoint_names]
            print("New keypoints:", new_keypoints)
            # Assign the modified values back to the Dataset
            ds = ds.assign_coords(keypoints=new_keypoints)

        # reorder the keypoints
        if ordered_keypoints is None or len(ordered_keypoints) == 0:
            print("No KPs to reorder. Skipping reordering step.")
        else:
            # Reorder the keypoints in the Dataset
            ds = ds.reindex(keypoints=ordered_keypoints)

        # save poses to dlc file format
        save_poses.to_dlc_file(ds, dest_path)

    # raise ValueError if path does not exist.
    else:
        raise ValueError(
            f"File '{fpath}' does not exist. "
            f"Please check the file path and try again."
        )


# %%
# The function below enables you to run the
# convert_inference_file function on a
# whole dataset to convert all the desired files
# modify this  file_location variable to point
# to the location of your pose_estimation files
# file_location = /path/to/dataset


def convert_all(file_location, ext=".slp"):
    # turn your string into a path object
    source_folder = pathlib.Path(file_location)

    # create a list of all files in your file_locatuon
    # that have the extension you specified
    fpaths = list(source_folder.rglob(f"*{ext}"))

    # run the convert_inference_file function on all
    # the files in the list
    for fpath in fpaths:
        convert_inference_file(fpath)
