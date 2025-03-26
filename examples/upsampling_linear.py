"""Script to upsample bounding boxes in time using linear interpolation
and correct the centroid position.
"""

# %%
import json  # Fixes E402 (module-level imports at the top)

import sleap_io as sio
from matplotlib import pyplot as plt

from movement import sample_data
from movement.filtering import interpolate_over_time
from movement.io import load_bboxes

# %%
dataset_dict = sample_data.fetch_dataset_paths(
    "VIA_single-crab_MOCA-crab-1.csv",
    with_video=True,  # Download associated video
)

file_path = dataset_dict["bboxes"]
print(file_path)

ds = load_bboxes.from_via_tracks_file(
    file_path, use_frame_numbers_from_file=True
)

# %%
print(ds)

# Inspecting associated video
video_path = dataset_dict["video"]
video = sio.load_video(video_path)
n_frames, height, width, channels = video.shape

print(f"Number of frames: {n_frames}")
print(f"Frame size: {width}x{height}")
print(f"Number of channels: {channels}")

print(ds.time)

# %%
ds_nan = ds.reindex(
    {"time": list(range(ds.time[-1].item() + 1))},
    method=None,  # Default
)

ds_interp = ds_nan.copy()

# Perform linear interpolation
for data_array_str in ["position", "shape"]:
    ds_interp[data_array_str] = interpolate_over_time(
        data=ds_interp[data_array_str],
        method="linear",
        max_gap=None,
        print_report=False,
    )

print(ds_interp)
print(ds_interp.time)

# Print original dataset for verification
print("Original dataset")
for t in [0, 5, 10]:
    print(f"Time {t}: {ds.position.sel(time=t).data}")

# Print upsampled dataset
print("Upsampled dataset")
for t in range(6):
    print(f"Time {t}: {ds_interp.position.sel(time=t).data}")


# %%
def plot_position_and_shape_xy_coords(ds_input, ds_filled, color_filled):
    """Compare the x and y coordinates of the position and shape arrays in
    time for the input and filled datasets.
    """
    fig, axs = plt.subplots(2, 2, figsize=(8, 6))
    space_coords = ["x", "y"]
    data_array_names = ["position", "shape"]

    for row, space_coord in enumerate(space_coords):
        for col, data_array in enumerate(data_array_names):
            ax = axs[row, col]
            ax.scatter(
                x=ds_input.time,
                y=ds_input[data_array].sel(
                    individuals="id_1", space=space_coord
                ),
                marker="o",
                color="black",
                label="original data",
            )
            ax.plot(
                ds_filled.time,
                ds_filled[data_array].sel(
                    individuals="id_1", space=space_coord
                ),
                marker=".",
                linewidth=1,
                color=color_filled,
                label="upsampled data",
            )
            ax.set_ylabel(f"{space_coord} (pixels)")
            if row == 0:
                ax.set_title(f"Bounding box {data_array}")
                if col == 1:
                    ax.legend()
            if row == 1:
                ax.set_xlabel("Time (frames)")

    fig.tight_layout()


plot_position_and_shape_xy_coords(
    ds, ds_filled=ds_interp, color_filled="tab:orange"
)

# Set FPS attribute for the dataset
ds_interp.attrs["fps"] = 24.0


# %%
def convert_to_via_format(ds, filename):
    """Convert Xarray dataset to VIA CSV format with correctly formatted
    bounding boxes.
    """
    df = ds.to_dataframe().reset_index()

    # Pivot data to organize x, y, width, height correctly
    df_pivot = df.pivot(
        index=["time"], columns="space", values=["position", "shape"]
    )

    # Flatten multi-index column names
    df_pivot.columns = ["x_center", "y_center", "width", "height"]
    df_pivot = df_pivot.reset_index()

    # Convert centroid back to top-left corner
    df_pivot["x"] = df_pivot["x_center"] - df_pivot["width"] / 2
    df_pivot["y"] = df_pivot["y_center"] - df_pivot["height"] / 2

    # Generate correct filenames
    df_pivot["filename"] = df_pivot["time"].apply(
        lambda t: f"/crab_1/{t:05d}.jpg"
    )

    # Add metadata fields
    df_pivot["file_size"] = 0
    df_pivot["file_attributes"] = json.dumps({"shot": 123})
    df_pivot["region_count"] = 1
    df_pivot["region_id"] = 0
    df_pivot["region_attributes"] = json.dumps({"track": 1})

    # Convert into VIA bounding box format
    df_pivot["region_shape_attributes"] = df_pivot.apply(
        lambda row: json.dumps(
            {
                "name": "rect",
                "x": row["x"],  # Corrected Top-Left Corner
                "y": row["y"],  # Corrected Top-Left Corner
                "width": row["width"],
                "height": row["height"],
            }
        ),
        axis=1,
    )

    # Keep only required columns
    df_final = df_pivot[
        [
            "filename",
            "file_size",
            "file_attributes",
            "region_count",
            "region_id",
            "region_shape_attributes",
            "region_attributes",
        ]
    ]

    # Save to CSV
    df_final.to_csv(filename, index=False)
    print(f"Saved: {filename}")


# Convert interpolated dataset and save
convert_to_via_format(ds_interp, "check_upsampled_dataset.csv")
