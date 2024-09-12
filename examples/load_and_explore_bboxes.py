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
from cycler import cycler
from matplotlib import pyplot as plt

from movement import sample_data
from movement.io import load_bboxes

# %%
# Select sample data file
# --------------------
# For the sake of this example, we will use the path to one of
# the sample datasets provided with ``movement``.

file_path = sample_data.fetch_dataset_paths(
    "VIA_multiple-crabs_5-frames_labels.csv"
)["bboxes"]
print(file_path)

# %%
# Read file as a `movement` dataset
# ----------------------------------
ds = load_bboxes.from_via_tracks_file(file_path)

# print some information about the dataset
print(ds)
print("-----")
print(f"Number of individuals: {ds.sizes['individuals']}")
print(f"Number of frames: {ds.sizes['time']}")


# %%
# The dataset contains bounding boxes for 86 individuals, tracked for
# 5 frames, in the xy plane.
#
# We can also see from the printout of the dataset that it contains
# three data arrays: ``position``, ``shape`` and ``confidence``.
#
# We will use these three arrays in the following sections to produce
# informative plots of the tracked trajectories
# %%
# Plot trajectories and color by individual
# -----------------------------------------

fig, ax = plt.subplots(1, 1)  # , figsize=(15, 15))

# add color cycler to axes
plt.rcParams["axes.prop_cycle"] = cycler(color=plt.get_cmap("tab10").colors)
# get the list of colors in the cycle
color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]


for id_idx, id_str in enumerate(ds["individuals"].data):
    ax.scatter(
        x=ds.position.sel(individuals=id_str, space="x").data,
        y=ds.position.sel(individuals=id_str, space="y").data,
        s=1,
        color=color_cycle[id_idx % len(color_cycle)],
    )
    # find first frame with non-nan x-coord
    start_frame = ds.time[
        ~ds.position.sel(individuals="id_1", space="y").isnull().data
    ][0]
    ax.text(
        x=ds.position.sel(
            time=start_frame, individuals=id_str, space="x"
        ).data,
        y=ds.position.sel(
            time=start_frame, individuals=id_str, space="y"
        ).data,
        s=str(id_str),
        horizontalalignment="center",
        color=color_cycle[id_idx % len(color_cycle)],
    )

ax.invert_yaxis()  # OJO!
# ax.set_ylim(0, 2160)
# ax.set_xlim(0, 4096)
ax.set_aspect("equal")
ax.set_xlabel("x (pixels)")
ax.set_ylabel("y (pixels)")
plt.show()

# %%
