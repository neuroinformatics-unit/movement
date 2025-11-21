"""Align videos using ROIs
==========================

Align two videos based on corresponding polygons (ROIs) defined in each video.
"""

# %%
# Imports
# -------

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from movement.roi import PolygonOfInterest
from movement.transforms import compute_homography_transform

# %%
# Define path to data
# -----------------------

data_dir = Path("/home/niko/Dropbox/NIU/data/Octagon")

target_roi_path = data_dir / "CameraColorTop_ROI.csv"
source_roi_path = data_dir / "CameraTop_ROI.csv"

assert target_roi_path.exists(), f"File not found: {target_roi_path}"
assert source_roi_path.exists(), f"File not found: {source_roi_path}"
# %%
# Load ROIs as polygon objects
# ----------------------------
# The ROIs are saved as napari Shapes layer format.

target_roi = pd.read_csv(target_roi_path)
source_roi = pd.read_csv(source_roi_path)


def roi_to_polygon(roi_df: pd.DataFrame, name: str) -> PolygonOfInterest:
    """Convert a ROI DataFrame to a PolygonOfInterest object."""
    # Extract points
    points = roi_df[["axis-1", "axis-2"]].values
    # To make a polygon, we must supply vertices as a list of (x, y) pairs
    point_list = [(x, y) for x, y in points]
    # Create polygon
    polygon = PolygonOfInterest(exterior_boundary=point_list)
    return polygon


target_polygon = roi_to_polygon(target_roi, name="Target")
source_polygon = roi_to_polygon(source_roi, name="Source")

# %%
# Plot ROIs
fig, ax = plt.subplots()
target_polygon.plot(ax=ax, color="blue", alpha=0.5, label="Target ROI")
source_polygon.plot(ax=ax, color="red", alpha=0.5, label="Source ROI")
ax.invert_yaxis()  # Flip the y-axis to match image coordinates
ax.legend()
plt.show()


# %%
# Align source to target using homography
# --------------------------------------
# Extract vertices

src_points = np.array(source_polygon.coords.xy).T  # Shape (N, 2)
dst_points = np.array(target_polygon.coords.xy).T  # Shape (N, 2)


transform = compute_homography_transform(src_points, dst_points)

print("Computed homography matrix:")
print(transform)
