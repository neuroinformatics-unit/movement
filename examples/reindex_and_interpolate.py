"""Reindex and interpolate bboxes tracks
===============================

Load and explore an example dataset of bounding boxes tracks.
"""

# %%
from movement import sample_data
from movement.filtering import interpolate_over_time
from movement.io import load_bboxes

# %%
# Select sample data file
# --------------------
# For the sake of this example, we will use the path to one of
# the sample datasets provided with ``movement``.

file_path = sample_data.fetch_dataset_paths("VIA_single-crab_MOCA-crab-1.csv")[
    "bboxes"
]
print(file_path)

ds = load_bboxes.from_via_tracks_file(
    file_path, use_frame_numbers_from_file=True
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Only 1 in 5 frames are labelled!
print(ds)
print(ds.time)
print(ds.position.data[:, 0, :])


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Extend the dataset to every frame by forward filling
ds_ff = ds.reindex(
    {"time": list(range(ds.time[-1].item()))},
    method="ffill",  # propagate last valid index value forward
)

print(ds_ff.position.data[:, 0, :])
print(ds_ff.shape.data[:, 0, :])

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Extend the dataset to every frame and fill empty values with nan
ds_nan = ds.reindex(
    {"time": list(range(ds.time[-1].item()))},
    method=None,  # default
)

print("Position data array:")
print(ds_nan.position.data[:11, 0, :])

print("Shape data array:")
print(ds_nan.shape.data[:11, 0, :])

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Linearly interpolate position and shape with nan

ds_interp = ds_nan.copy()

for data_array_str in ["position", "shape"]:
    ds_interp[data_array_str] = interpolate_over_time(
        data=ds_interp[data_array_str],
        method="linear",
        max_gap=None,
        print_report=False,
    )

print("Position data array:")
print(ds_interp.position.data[:11, 0, :])

print("Shape data array:")
print(ds_interp.shape.data[:11, 0, :])

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Export as csv file
