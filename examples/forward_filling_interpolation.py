"""Forward Filling Interpolation Script

This script performs forward filling interpolation on bounding box data
and saves the results in VIA format.
"""

# All imports should be at the top (Fixes E402)
import json

import sleap_io as sio
from matplotlib import pyplot as plt

from movement import sample_data
from movement.io import load_bboxes

# Fetch dataset
dataset_dict = sample_data.fetch_dataset_paths(
    "VIA_single-crab_MOCA-crab-1.csv",
    with_video=True,  # download associated video
)
file_path = dataset_dict["bboxes"]
print(file_path)

ds = load_bboxes.from_via_tracks_file(
    file_path, use_frame_numbers_from_file=True
)


#  Inspect associated video
video_path = dataset_dict["video"]
video = sio.load_video(video_path)
n_frames, height, width, channels = video.shape
print(f"Number of frames: {n_frames}")
print(f"Frame size: {width}x{height}")
print(f"Number of channels: {channels}")


# Apply forward filling interpolation
ds_ff = ds.reindex(
    {"time": list(range(ds.time[-1].item() + 1))},
    method="ffill",  # propagate last valid index value forward
)


def plot_position_and_shape_xy_coords(ds_input_data, ds_filled, color_filled):
    """Compare the x and y coordinates of the position and shape arrays in time
    for the input and filled datasets.
    """
    fig, axs = plt.subplots(2, 2, figsize=(8, 6))
    for row in range(axs.shape[0]):
        space_coord = ["x", "y"][row]
        for col in range(axs.shape[1]):
            ax = axs[row, col]
            data_array_str = ["position", "shape"][col]

            # plot original data
            ax.scatter(
                x=ds_input_data.time,
                y=ds_input_data[data_array_str].sel(
                    individuals="id_1", space=space_coord
                ),
                marker="o",
                color="black",
                label="original data",
            )

            # plot forward filled data
            ax.plot(
                ds_filled.time,
                ds_filled[data_array_str].sel(
                    individuals="id_1", space=space_coord
                ),
                marker=".",
                linewidth=1,
                color=color_filled,
                label="upsampled data",
            )

            # set axes labels and legend
            ax.set_ylabel(f"{space_coord} (pixels)")
            if row == 0:
                ax.set_title(f"Bounding box {data_array_str}")
                if col == 1:
                    ax.legend()
            if row == 1:
                ax.set_xlabel("time (frames)")

    fig.tight_layout()


# Print sample data
print("Original dataset")
print(ds.position.sel(time=0).data)
print(ds.position.sel(time=5).data)
print(ds.position.sel(time=10).data)


print("Upsampled dataset")
for i in range(6):
    print(ds_ff.position.sel(time=i).data)


# Plot the results
plot_position_and_shape_xy_coords(
    ds, ds_filled=ds_ff, color_filled="tab:green"
)


# Convert to VIA format and save
def convert_to_via_format(ds, filename):
    """Convert Xarray dataset to VIA CSV format with
    correctly formatted bounding boxes.
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


convert_to_via_format(ds_ff, "check_ffupsampled_dataset.csv")
