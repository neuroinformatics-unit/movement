"""Load and explore bboxes tracks
===============================

Load and explore an example dataset of bounding boxes tracks.
"""

# %%
# Imports
# -------
# For interactive plots: install ipympl with `pip install ipympl` and uncomment
# the following line in your notebook
# %matplotlib widget
from pathlib import Path

from cycler import cycler
from matplotlib import pyplot as plt

from movement.io import load_bboxes

# %%
# Select sample data file
# --------------------
# - Download data from https://www.robots.ox.ac.uk/~vgg/software/via/docs/face_track_annotation.html
# - Edit name of files to include `frame_`

# %%
# Read file as a `movement` dataset
# ----------------------------------
file_path = (
    Path.home()
    / "Downloads"
    / "face_track_annotation"
    / "data"
    / "sherlock_ep01_tracks_FRCNN_SM.csv"
)
ds = load_bboxes.from_via_tracks_file(
    str(file_path), use_frame_numbers_from_file=True
)

# restrict to first shot only
start_end_frames_shot_1 = (11384, 11586)
frames_shot_1 = list(
    range(start_end_frames_shot_1[0], start_end_frames_shot_1[1], 1)
)
ds = ds.sel(time=frames_shot_1).copy()  # I think I need copy?

# remove individuals whose position is nan for all frames in the shot
# bool_individuals_all_nan = np.all(np.isnan(ds.position.data), axis=(0, 2))
# ds = ds.drop_sel(individuals=ds.individuals.data[bool_individuals_all_nan])
ds = ds.dropna(dim="individuals", how="all")

# print some information about the dataset
print(ds)
print("-----")
print(f"Number of individuals: {ds.sizes['individuals']}")
print(f"Number of frames: {ds.sizes['time']}")


# %%
# The reduced dataset contains bounding boxes for 2 individuals, tracked for
# 202 frames, in the xy plane.
#
# We can also see from the printout of the dataset that it contains
# three data arrays: ``position``, ``shape`` and ``confidence``.
# %%
# Plot trajectories of first shot and color by individual
# -----------------------------------------

fig, ax = plt.subplots(1, 1)

# add color cycler to axes
plt.rcParams["axes.prop_cycle"] = cycler(color=plt.get_cmap("tab10").colors)
# get the list of colors in the cycle
color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]


frame_number = frames_shot_1[0]
img = plt.imread(
    str(file_path.parents[1] / "images" / f"{frame_number:08}.jpg")
)

for id_idx, id_str in enumerate(ds["individuals"].data):
    # plot first frame
    ax.imshow(img)

    past_frames = [f for f in frames_shot_1 if f <= frame_number]
    future_frames = [f for f in frames_shot_1 if f > frame_number]

    # plot past position of centroid in grey
    ax.scatter(
        x=ds.position.sel(
            individuals=id_str, time=past_frames, space="x"
        ).data,
        y=ds.position.sel(
            individuals=id_str, time=past_frames, space="y"
        ).data,
        s=1,
        color="grey",
    )
    # plot future trajectories of centroids in color
    ax.scatter(
        x=ds.position.sel(
            individuals=id_str, time=future_frames, space="x"
        ).data,
        y=ds.position.sel(
            individuals=id_str, time=future_frames, space="y"
        ).data,
        s=1,
        color=color_cycle[id_idx % len(color_cycle)],
    )
    # plot bbox in this frame
    # ATT! currently position is the top left corner of bbox
    # need to uncomment the line below if position loaded is centroid
    top_left_corner = (
        ds.position.sel(individuals=id_str, time=frame_number).data
        - ds.shape.sel(individuals=id_str, time=frame_number).data / 2
    )
    bbox = plt.Rectangle(
        xy=tuple(top_left_corner),
        width=ds.shape.sel(
            individuals=id_str, time=frame_number, space="x"
        ).data,
        height=ds.shape.sel(
            individuals=id_str, time=frame_number, space="y"
        ).data,
        edgecolor=color_cycle[id_idx % len(color_cycle)],
        facecolor="none",  # transparent fill
        linewidth=1.5,
    )
    ax.add_patch(bbox)


# ax.legend(ds["individuals"].data, bbox_to_anchor=(1.0, 1.0))
# ax.invert_yaxis()
ax.set_aspect("equal")
ax.set_xlabel("x (pixels)")
ax.set_ylabel("y (pixels)")
ax.set_title(f"Sherlock - shot 1, frame {frame_number}")
plt.show()

# %%
