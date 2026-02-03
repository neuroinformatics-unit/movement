# %%
from pathlib import Path

import matplotlib.pyplot as plt
import xarray as xr
import zarr

from movement.io import load_bboxes
from movement.plots import plot_centroid_trajectory, plot_occupancy

# Hide attributes globally
xr.set_options(display_expand_attrs=False)

# %%
# %matplotlib widget

# %%
# Input data
trials_dir = Path(
    "/Users/sofia/arc/project_Zoo_crabs/loops_tracking_above_10th_percentile_slurm_1825237_SAMPLE"
)


# List of trial files
list_files = sorted(list(trials_dir.glob("*.csv")))

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Save combined dataset as zarr
# build up the zarr file incrementally

output_zarr_file = trials_dir / "all_trials.zarr"

for i, file_path in enumerate(list_files):
    # Load single dataset
    trial_ds = load_bboxes.from_via_tracks_file(file_path)

    # Add inbound/outbound array along time coordinate

    # Expand dimensions: video, loop_id
    # NOTE: I think I would need to make all trials same n frames
    trial_ds = trial_ds.expand_dims({"trial": [i]})  # save as string?

    # Define chunk
    trial_ds = trial_ds.chunk({"time": 1000})

    # Save chunk
    # Append mode (region write) for subsequent trials
    if i == 0:
        trial_ds.to_zarr(output_zarr_file, mode="w")
    else:
        trial_ds.to_zarr(output_zarr_file, mode="a", append_dim="trials")


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Hierarchical group approach

# Create a zarr group structure
# x - fail if exists
# w - overwrite if exists
root = zarr.open_group("all_trials_hierarchical.zarr", mode="w-")

for i, file_path in enumerate(list_files[:3]):
    print(f"Processing file {file_path.name}")

    # load
    trial_ds = load_bboxes.from_via_tracks_file(file_path)

    # add additional data
    # - add inbound / outbound along time coordinate
    # - add video name
    # - add clip ID (as coordinate?)
    # - add fps
    # - add frame start / end

    # hierarchy
    # - date
    # - video ID
    # - loop ID

    # extract metadata from filename
    video_name, loop_part = file_path.name.rsplit("-", 1)
    # e.g. '04.09.2023-01-Right'
    date, video_id = video_name.split("-", 1)
    clip_id = loop_part.split("_tracks.csv")[0]

    # chunk along time
    trial_ds = trial_ds.chunk({"time": 1000})

    # Save each trial as a separate group in the structure
    # e.g.'04.09.2023/01-Right/Loop05'
    group_path = f"{date}/{video_id}/{clip_id}"
    trial_ds.to_zarr(root.store, group=group_path)

# %%
print(root.tree())


# %%%%%%%%%%%%%%%
# Later, load as an xarray DataTree
import xarray as xr

dt = xr.open_datatree("all_trials_hierarchical.zarr", engine="zarr")

print(dt)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Inspect structure

print(f"Depth: {dt.depth}")  # 3

print(f"Values at 1st level: {list(dt.children.keys())}")
print(f"Values at 2nd level: {list(dt['04.09.2023'].children.keys())}")
print(
    f"Values at 3rd level: {list(dt['04.09.2023/01-Right'].children.keys())}"
)

# %%
print("Flat list of all paths:")
print(*dt.groups, sep="\n")

print("List of leaf paths only:")
print(*[node.path for node in dt.leaves], sep="\n")

print("Inspect dimension of leaves:")
for node in dt.leaves:
    if node.has_data:
        print(
            f"{node.path}: dims={dict(node.sizes)}, "
            f"vars={list(node.data_vars)}"
        )


# %%%%%%%%%%%%%%%%%%%%%%%%%%%
# Access a specific "Loop" clip
# .ds returns view, .to_dataset() returns dataset
ds_loop09 = dt["04.09.2023/01-Right/Loop09"].to_dataset()


# %%
# plot centroid trajectory of all individuals
fig, ax = plt.subplots()
colors = plt.cm.tab20.colors
for i, ind in enumerate(ds_loop09.individuals):
    plot_centroid_trajectory(
        ds_loop09.position,
        individual=ind,
        ax=ax,
        c=colors[i % len(colors)],
    )
ax.invert_yaxis()
ax.set_xlabel("x (pixels)")
ax.set_ylabel("y (pixels)")
ax.set_aspect("equal")
ax.set_title(
    Path(ds_loop09.attrs["source_file"]).name.split("_tracks.csv")[0]
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Compute occupancy for all clips from a video
# (we assume camera does not move within one day)


# Return nodes with path matching the pattern
leaves_pattern = "04.09.2023/01-Right/*"
dt_one_video = dt.match(leaves_pattern)

# note: still three levels
print(dt_one_video.depth)

# Prepare data for plot
# concatenate all position arrays
position_arrays = []
for ds_leaf in dt_one_video.leaves:
    if ds_leaf.has_data:
        position_arrays.append(ds_leaf.position)

position_combined = xr.concat(
    position_arrays,
    dim="loops",
    join="outer"
)

# flatten to just time and space coords
position_flat = position_combined.stack(
    flat_dim = ("loops", "time", "individuals")
).dropna("flat_dim")


# plot for all clips in video
fig, ax = plt.subplots()
ax.hist2d(
    position_flat.sel(space="x").values,
    position_flat.sel(space="y").values,
    bins=[100,50], cmap="viridis"
)
ax.invert_yaxis()
ax.set_xlabel("x (pixels)")
ax.set_ylabel("y (pixels)")
ax.set_aspect("equal")
ax.set_title(leaves_pattern)


# %%
# Find all nodes matching a pattern
# e.g. first video of every day
dt.match("*/01-Right/*")


# %%
# Iterate over dates
for dt_date in dt.children.values():
    for dt_video in dt_date.children.values():
        for dt_loop in dt_video.children.values():
            # process each loop clip
            pass
