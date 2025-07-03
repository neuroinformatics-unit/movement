"""Time spent in regions of interest
====================================

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
import pandas as pd
from matplotlib import pyplot as plt

from movement import sample_data
from movement.filtering import filter_by_confidence
from movement.plots import plot_occupancy
from movement.roi import PolygonOfInterest, compute_region_occupancy

# %%
# Load data
# ---------
# The elevated plus maze dataset is provided as part of ``movement``'s
# sample data. We load the dataset and inspect its contents.

ds = sample_data.fetch_dataset("DLC_single-mouse_EPM.predictions.h5")
print(ds)
print("-" * 80)
print(f"Individuals: {ds.individuals.values}")
print(f"Keypoints: {ds.keypoints.values}")

# %%
# We will do some basic filtering, i.e. drop points with low confidence.

position = filter_by_confidence(
    ds.position, ds.confidence, threshold=0.95, print_report=True
)

# %%
# Plot occupancy
# --------------
# A quick way to get an impression about the relative time spent in
# different regions of the maze is to use the
# :func:`plot_occupancy() <movement.plots.occupancy.plot_occupancy>` function.
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


fig, ax = plt.subplots()
ax.imshow(image)  # Show the image

# Plot the occupancy 2D histogram for the centroid of all keypoints
fig, ax, hist_data = plot_occupancy(
    da=ds.position,
    ax=ax,
    alpha=0.8,
    bins=bins,
    cmin=10,  # Set the minimum shown count
    norm="log",
)

ax.set_title("Occupancy heatmap")
# Set the axis limits to match the image
ax.set_xlim(0, width)
ax.set_ylim(height, 0)
ax.collections[0].colorbar.set_label("# frames")

# %%
# Define ROIs in napari
# ---------------------
# We can define regions of interest (ROIs) as polygons in pixel coordinates.
#
# To do that we will first launch the
# `movement GUI <https://movement.neuroinformatics.dev/user_guide/gui.html>`_
# and load the video (or an image extracted from it) as a background layer,
# as described in the GUI guide.
#
# Then, you can create a new ``Shapes`` layer and draw polygons around the
# regions of interest. See the corresponding
# `napari guide <https://napari.org/dev/howtos/layers/shapes.html>`_
# for more information on how to create and edit shapes in napari.
#
# For the purposes of this example, we want to draw around the following
# regions, in that order:
#
# - Open arm (left)
# - Open arm (right)
# - Closed arm (bottom)
# - Closed arm (top)
# - Central square
#
# If you are running this example in a Jupyter notebook on your local machine,
# you can run the following code to launch the `napari` with the frame
# loaded, and add an empty Shapes layer so you are all set to draw the ROIs.
#
# .. code-block:: python
#
#     import napari
#     from napari.layers import Shapes
#
#     viewer = napari.Viewer()
#     viewer.open(ds.frame_path)
#     viewer.layers[0].name = "EPM frame"
#
#     shapes_layer = Shapes(name="EPM ROIs")
#     viewer.add_layer(shapes_layer)
#     shapes_layer.mode = "ADD_POLYGON"
#
#
# One you've drawn the above 5 polygons, you can select the
# Shapes layer in the napari GUI and save it to a .csv file
# via ``File > Save Selected Layers...``. At any later point, you can
# load the ROIs from the .csv file by dragging and dropping it into the
# napari window, or by using the ``File > Open...`` menu.

# %%
# Load ROIs from file
# -------------------
# We can load the ROIs from the .csv file saved in napari.
# Let's see what the loaded dataframe looks like.

rois_df = pd.read_csv("EPM_ROIs.csv")
rois_df.head()

# %%
# The ``index`` column contains the index of the ROI, which corresponds to
# the order in which the ROIs were drawn in napari. The ``shape-type``
# column should be all ``polygon`` in our case. The ``vertex-index``
# column contains the index of the vertex (points) in each polygon, also
# in the order they were drawn in napari, with the first and last points
# being the same (i.e. the polygon is closed). The ``axis-0`` and
# ``axis-1`` columns contain the y and x coordinates of the vertices,
# respectively, in pixel coordinates.
#
# Let's re-format the dataframe to make it easier to work with.

# Add ROI names based on the order they were drawn
roi_names = [
    "open_arm_left",
    "open_arm_right",
    "closed_arm_bottom",
    "closed_arm_top",
    "central_square",
]
rois_df["name"] = rois_df["index"].apply(lambda x: roi_names[x])

# Rename the columns for clarity
rois_df.rename(
    columns={
        "axis-0": "y",
        "axis-1": "x",
        "index": "roi-index",
    },
    inplace=True,
)
# Set the index to roi-index, then vertex-index
rois_df.set_index(["roi-index", "vertex-index"], inplace=True)
rois_df.head()

# %%
# Visualise ROIs
# --------------
# We will now convert each napari shape into a ``movement``
# :class:`movement.roi.PolygonOfInterest` object.
# To construct each polygon, we need to pass a list of (x, y) tuples
# representing the vertices of the polygon, and the name of the ROI.
# Note that we'll exclude the last point in each polygon, as ``movement``
# knows how to close the polygon automatically.

rois = []
for _, roi_df in rois_df.groupby("roi-index"):
    # Get the name of the ROI
    roi_name = roi_df["name"].iloc[0]
    # Get the vertices of the polygon as a list of (x, y) tuples
    vertices = list(zip(roi_df["x"][:-1], roi_df["y"][:-1], strict=True))
    # Create a PolygonOfInterest object
    rois.append(PolygonOfInterest(exterior_boundary=vertices, name=roi_name))

# %%
# Now that we have the ROIs as
# :class:`PolygonOfInterest <movement.roi.polygon.PolygonOfInterest>` objects,
# we can make use of lots of useful methods they provide.
# For example, we can plot the ROIs on top of the image of the maze using
# the :meth:`plot() <movement.roi.polygon.PolygonOfInterest.plot>` method.

fig, ax = plt.subplots()
ax.imshow(image)

colors = plt.cm.Dark2.colors
for i, roi in enumerate(rois):
    roi.plot(ax, facecolor=colors[i], alpha=0.5)

ax.legend()

# %%
# Compute time spent in each ROI
# ------------------------------
# Now we can compute the time spent in each ROI using the
# :func:`compute_region_occupancy() \
# <movement.roi.conditions.compute_region_occupancy>` function.

occupancy = compute_region_occupancy(
    ds.position,
    regions=rois,
)
occupancy

# %%
# The occupancy data is an :class:`xarray.DataArray` with similar dimensions
# to ``ds.position``, but the ``space`` dimension is replaced by ``region``.
# It contains boolean values indicating if a keypoint was inside a region
# at a given time.
#
# To determine if the mouse as a whole was in a region, we can use different
# approaches, such as:
#
# - Checking if the centroid of all keypoints is in the region.
# - Checking if a specific keypoint, e.g., `tailbase`, is in the region.
# - Checking if all specific keypoints are in the region.
#
# Below are examples of these methods.

# Check if the centroid of all keypoints is in the region

centroid_in_region = compute_region_occupancy(
    ds.position.mean(dim="keypoints"),
    regions=rois,
)

# Count number of frames in each region
frames_in_region = centroid_in_region.sum(dim="time")
# Convert to percentage of total frames
frames_in_region = 100 * (frames_in_region / centroid_in_region.sizes["time"])

frames_in_region.plot.step(
    x="region",
    where="mid",
)
plt.ylabel("% time spent in region")
plt.xticks(rotation=30)

# %%
