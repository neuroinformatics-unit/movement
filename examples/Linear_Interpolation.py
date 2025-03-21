# %%
import sleap_io as sio
from matplotlib import pyplot as plt

from movement import sample_data
from movement.filtering import interpolate_over_time
from movement.io import load_bboxes

# %%
dataset_dict = sample_data.fetch_dataset_paths(
    "VIA_single-crab_MOCA-crab-1.csv",
    with_video=True,  # download associated video
)

file_path = dataset_dict["bboxes"]
print(file_path)

ds = load_bboxes.from_via_tracks_file(
    file_path, use_frame_numbers_from_file=True
)


# %%
print(ds)

# %% [markdown]
# Inspecting associated video

# %%
video_path = dataset_dict["video"]

video = sio.load_video(video_path)
n_frames, height, width, channels = video.shape

print(f"Number of frames: {n_frames}")
print(f"Frame size: {width}x{height}")
print(f"Number of channels: {channels}")

# %%
print(ds.time)

# %%
ds_nan = ds.reindex(
    {"time": list(range(ds.time[-1].item() + 1))},
    method=None,  # default
)

# %%
ds_interp = ds_nan.copy()

# %%
for data_array_str in ["position", "shape"]:
    ds_interp[data_array_str] = interpolate_over_time(
        data=ds_interp[data_array_str],
        method="linear",
        max_gap=None,
        print_report=False,
    )

# %%
print(ds_interp)

# %%
print(ds_interp.time)

# %%
print(ds.position.sel(time=0).data)
print(ds.position.sel(time=5).data)
print(ds.position.sel(time=10).data)


# %%
print(ds_interp.position.sel(time=0).data)
print(ds_interp.position.sel(time=1).data)
print(ds_interp.position.sel(time=2).data)
print(ds_interp.position.sel(time=3).data)
print(ds_interp.position.sel(time=4).data)
print(ds_interp.position.sel(time=5).data)


# %%
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


# %%


plot_position_and_shape_xy_coords(
    ds, ds_filled=ds_interp, color_filled="tab:orange"
)

# %%
ds_interp.attrs["fps"] = 24.0  # Assign a meaningful value or another number

# %%
import pandas as pd

# Convert ds_interp to a Pandas DataFrame
df_interp = ds_interp.to_dataframe().reset_index()

# Save as CSV (similar to the original format)
df_interp.to_csv("upsampled_dataset.csv", index=False)


# %%

# Convert ds_interp to a Pandas DataFrame
df = ds.to_dataframe().reset_index()

# Save as CSV (similar to the original format)
df.to_csv("nonupdataset.csv", index=False)

# %%
import json


def convert_to_via_format(df, filename):
    """Convert the given dataframe into VIA annotation format and save as CSV.

    Parameters
    ----------
    - df: Pandas DataFrame containing movement dataset (original or interpolated)
    - filename: Output CSV file name

    """
    # Construct filename from frame numbers
    df["filename"] = df["time"].apply(lambda t: f"/crab_1/{t:05d}.jpg")

    # Add static metadata fields
    df["file_size"] = 0  # Placeholder value
    df["file_attributes"] = json.dumps({"shot": 123})
    df["region_count"] = 1
    df["region_id"] = 0
    df["region_attributes"] = json.dumps({"track": 1})  # Assume one track

    # Pivot data to get 'x', 'y', 'width', and 'height' in the same row
    df_pivot = df.pivot(
        index=["time", "filename"],
        columns="space",
        values=["position", "shape"],
    )

    # Rename columns for clarity
    df_pivot.columns = ["x", "y", "width", "height"]
    df_pivot = df_pivot.reset_index()

    # Create a single bounding box JSON format
    df_pivot["region_shape_attributes"] = df_pivot.apply(
        lambda row: json.dumps(
            {
                "name": "rect",
                "x": row["x"],
                "y": row["y"],
                "width": row["width"],
                "height": row["height"],
            }
        ),
        axis=1,
    )

    # Add static metadata
    df_pivot["file_size"] = 0
    df_pivot["file_attributes"] = json.dumps({"shot": 123})
    df_pivot["region_count"] = 1
    df_pivot["region_id"] = 0
    df_pivot["region_attributes"] = json.dumps({"track": 1})

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


# Convert xarray datasets to DataFrames
df = ds.to_dataframe().reset_index()
df_interp = ds_interp.to_dataframe().reset_index()

# Process both datasets
convert_to_via_format(df, "nonupdataset.csv")  # Save original dataset
convert_to_via_format(
    df_interp, "upsampled_dataset.csv"
)  # Save interpolated dataset


# %%

df_original = pd.read_csv("nonupdataset.csv")
df_new = pd.read_csv("final_upsampled_dataset.csv")

print(df_original.head(3))  # Check column names
print(df_new.head(10))  # Should match original format


# %%
