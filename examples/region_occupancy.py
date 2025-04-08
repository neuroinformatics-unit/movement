"""Compute time spent in regions of interest
============================================

Define regions of interest and compute the time spent in each region.
"""

# %%
# Motivation
# ----------
# In this example we will work with a dataset of a mouse navigating
# an [elevated plus maze](https://en.wikipedia.org/wiki/Elevated_plus_maze),
# which consists of two open and two closed arms. Because of the
# general aversion of mice to open spaces, we expect mice to prefer the
# closed arms of the maze. Therefore, the proportion of time spent in
# open/closed arms is often used as a measure of anxiety-like behaviour
# in mice, i.e. the more time spent in the open arms, the less anxious the
# mouse is.

# %%
# Imports
# -------
import numpy as np
import seaborn as sns
import sleap_io as sio
import xarray as xr
import yaml
from matplotlib import pyplot as plt

from movement import sample_data
from movement.filtering import (
    filter_by_confidence,
    interpolate_over_time,
    rolling_filter,
)
from movement.kinematics import compute_forward_vector_angle
from movement.plots import plot_centroid_trajectory, plot_occupancy
from movement.roi import PolygonOfInterest, compute_region_occupancy

# Set the style for the plots
sns.set_context("poster")
sns.set_style("ticks")

# %%
# Load data
# ---------
# The elevated plus maze dataset is provided as part of ``movement``'s
# sample data. We load the dataset and inspect its contents.

ds = sample_data.fetch_dataset(
    "DLC_single-mouse_EPM.predictions.h5", with_video=True
)
print(ds)
print("-----------------------------")
print(f"Individuals: {ds.individuals.values}")
print(f"Keypoints: {ds.keypoints.values}")

# %%
# Do some basic filtering
# -----------------------
# We will drop points with low confidence.

position_high_confidence = filter_by_confidence(
    ds.position,
    ds.confidence,
    threshold=0.99,
)

# Next, we will apply a rolling mean filter to smooth the data.
position_smoothed = rolling_filter(
    position_high_confidence,
    window=9,
    statistic="median",
    min_periods=3,
)

# Finally, we will interpolate over time to fill in gaps smaller than 5 frames.
position_interpolated = interpolate_over_time(
    position_smoothed,
    method="linear",
    max_gap=5,
)

# Concatenate raw and clean data into a single array
positions = xr.concat([ds.position, position_interpolated], "data")
positions.coords["data"] = [
    "raw",
    "clean",
]

# %%

# Select data to plot
selection = dict(
    keypoints="snout",
    individuals="individual_0",
    space="y",
    time=slice(100, 300),
)

# Let's plot the raw vs the filtered data.
positions_sel = positions.sel(**selection)
positions_sel.plot.line(
    x="time",
    hue="data",
    aspect=4,
    size=4,
)
plt.xlim(positions_sel.time.min(), positions_sel.time.max())
plt.ylabel("x position (pixels)")
plt.xlabel("time (s)")
plt.title("Data cleaning")
legend = plt.gca().get_legend()
legend.set_title("")
legend.set_loc("lower left")

plt.tight_layout()
plt.savefig("raw_vs_clean.png", dpi=300)

# %%
# Plot occupancy
# --------------
# A quick way to get an impression about the relative time spent in
# different regions of the maze is to use the
# :func:`movement.plots.plot_occupancy` function.
# By default, this function will the occupancy of the centroid
# of all available keypoints, for the first individual
# in the dataset (in this case, the only individual).


# Load the frame and plo
image = plt.imread(ds.frame_path)
height, width, channel = image.shape

# Construct bins that cover the entire image
bin_pix = 30  # pixels
bins = [
    np.arange(0, width + bin_pix, bin_pix),
    np.arange(0, height + bin_pix, bin_pix),
]


fig, ax = plt.subplots(figsize=(8, 5.5))
ax.imshow(image.mean(axis=2), cmap="gray")  # Show the image in grayscale

# Plot the occupancy 2D histogram for the centroid of all keypoints
fig, ax, hist_data = plot_occupancy(
    da=position_interpolated,
    ax=ax,
    alpha=0.8,
    bins=bins,
    cmin=10,  # Set the minimum shown count
    norm="log",
    cmap="turbo",
)

ax.set_title("Occupancy heatmap")
# Set the axis limits to match the image
ax.set_xlim(0, width)
ax.set_ylim(height, 0)
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_xticks([])
ax.set_yticks([])
ax.collections[0].colorbar.set_label("# frames")

plt.tight_layout()
plt.savefig("occupancy_heatmap.png", dpi=300)

# %%
# Load the video frames
# ---------------------
video = sio.load_video(ds.video_path)
frames, width, height, channels = video.shape

# %%
# Plot the centroid trajectory

# Select time period to plot
time_window = slice(240, 250)
last_frame_idx = int(250 * ds.fps)
last_frame = video[last_frame_idx]

fig, ax = plt.subplots(figsize=(8, 5.5))
ax.imshow(last_frame.mean(axis=2), cmap="gray")  # Show the image in grayscale
# Plot the trajectory of the centroid of selected keypoints
plot_centroid_trajectory(
    position_interpolated.sel(time=time_window),
    individual="individual_0",
    keypoints=["centre", "tailbase", "left_ear", "right_ear"],
    ax=ax,
    alpha=0.5,
    cmap="viridis",
)
plt.title("Centroid trajectory")
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_xticks([])
ax.set_yticks([])
# Change the cbar label
cbar = ax.collections[0].colorbar
cbar.set_label("time (s)")

plt.tight_layout()
plt.savefig("centroid_trajectory.png", dpi=300)


# %%
# Let's define some regions of interest (ROIs)
# --------------------------------------------
# We will use the :mod:`movement.roi` module to define the ROIs.
# as :class:`movement.roi.PolygonOfInterest` objects.

# Let's load the ROIs from a yaml file.
rois: list[PolygonOfInterest] = []
with open("EPM_rois.yaml") as file:
    for roi_name, roi_coords in yaml.safe_load(file).items():
        rois.append(PolygonOfInterest(roi_coords, name=roi_name))
        print(f"ROI: {roi_name}")

# %%
# Compute occupancy in ROIs
# ------------------------------
# We will use the :func:`movement.roi.compute_region_occupancy` function
# which will return an boolean array of shape

roi_occupancy = compute_region_occupancy(
    position_interpolated,
    rois,
)

# %%
# We will consider that the mouse was in a region if all of the
# following keypoints were in the region:
# - centre
# - tailbase

mouse_in_roi = roi_occupancy.sel(
    keypoints=["centre", "tailbase"],
    individuals="individual_0",
).all(dim="keypoints")

# Let's sum the time spent in each region
# and plot the results.

frames_in_roi = mouse_in_roi.sum(dim="time")
# Convert to % of time spent in the region
pct_time_in_roi = frames_in_roi / mouse_in_roi.sizes["time"] * 100

# Let's plot the ROIs on top of the EPM maze.

fig, ax = plt.subplots(figsize=(6.3, 5.5))

# Convert the image to grayscale
image_gray = np.mean(image, axis=2)
ax.imshow(image_gray, cmap="gray")  # Show the image in grayscale

edge_palette = plt.get_cmap("Dark2", len(rois)).colors
fill_palette = plt.get_cmap("Set2", len(rois)).colors
for i, roi in enumerate(rois):
    roi.plot(
        ax=ax,
        edgecolor=edge_palette[i],
        facecolor=fill_palette[i],
        alpha=0.6,
        label=f"{pct_time_in_roi[i]:.0f}%",
        lw=2,
    )
    roi_centroid = np.array(roi.coords).mean(axis=0)
    ax.text(
        roi_centroid[0],
        roi_centroid[1],
        f"{pct_time_in_roi[i]:.0f}%",
        color="black",
        fontsize=12,
        ha="center",
        va="center",
    )
# ax.set_xlim(0, width)
# ax.set_ylim(height, 0)
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("Time spent in ROIs")

plt.tight_layout()
plt.savefig("rois.png", dpi=300)


# %%
# Compute the head direction angle
# --------------------------------

# %%
head_angle = compute_forward_vector_angle(
    position_interpolated.sel(individuals="individual_0"),
    left_keypoint="left_ear",
    right_keypoint="right_ear",
    # Optional parameters:
    reference_vector=(1, 0),  # positive x-axis
    camera_view="top_down",
    in_degrees=False,  # set to True for degrees
)
print(head_angle)


# %%
def plot_polar_histogram(da, bin_width_deg=15, ax=None):
    """Plot a polar histogram of the data in the given DataArray.

    Parameters
    ----------
    da : xarray.DataArray
        A DataArray containing angle data in radians.
    bin_width_deg : int, optional
        Width of the bins in degrees.
    ax : matplotlib.axes.Axes, optional
        The axes on which to plot the histogram.

    """
    n_bins = int(360 / bin_width_deg)

    if ax is None:
        fig, ax = plt.subplots(  # initialise figure with polar projection
            1, 1, figsize=(5, 5), subplot_kw={"projection": "polar"}
        )
    else:
        fig = ax.figure  # or use the provided axes

    # plot histogram using xarray's built-in histogram function
    da.plot.hist(
        bins=np.linspace(-np.pi, np.pi, n_bins + 1),
        ax=ax,
        density=True,
    )

    # axes settings
    ax.set_theta_direction(-1)  # theta increases in clockwise direction
    ax.set_theta_offset(0)  # set zero at the right
    ax.set_xlabel("")  # remove default x-label from xarray's plot.hist()

    # set xticks to match the phi values in degrees
    n_xtick_edges = 9
    ax.set_xticks(np.linspace(0, 2 * np.pi, n_xtick_edges)[:-1])
    xticks_in_deg = (
        list(range(0, 180 + 45, 45)) + list(range(0, -180, -45))[-1:0:-1]
    )
    ax.set_xticklabels([str(t) + "\N{DEGREE SIGN}" for t in xticks_in_deg])

    return fig, ax


# %%
# Plot the head direction angle
fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={"projection": "polar"})
plot_polar_histogram(
    head_angle.sel(time=slice(200, 300)),
    bin_width_deg=15,
    ax=ax,
)
ax.set_ylim(0, 0.25)  # force same y-scale (density) for both plots
ax.set_yticks([0, 0.1, 0.2])
ax.set_yticklabels(["", "", ""])
ax.set_title(" ")
ax.set_title("Head direction angle", pad=45)
fig.subplots_adjust(
    left=0.2,
    right=0.8,
    top=0.8,
    bottom=0.2,
)
plt.tight_layout()
plt.savefig("head_direction_angle.png", dpi=300)


# %%
