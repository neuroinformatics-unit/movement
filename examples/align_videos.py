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

from movement.io import load_poses, save_poses
from movement.roi import PolygonOfInterest
from movement.transforms import (
    compute_homography_transform,
    transform_points_homography,
)

# %%
# Define path to data
# -----------------------

data_dir = Path("/Users/nsirmpilatze/Dropbox (Personal)/NIU/data/Octagon")

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
    polygon = PolygonOfInterest(exterior_boundary=point_list, name=name)
    return polygon


target_polygon = roi_to_polygon(target_roi, name="Target")
source_polygon = roi_to_polygon(source_roi, name="Source")

# %%
# Plot ROIs
fig, ax = plt.subplots()
target_polygon.plot(ax=ax, facecolor="m", alpha=0.5, label="Target ROI")
source_polygon.plot(ax=ax, facecolor="c", alpha=0.5, label="Source ROI")
ax.invert_yaxis()  # Flip the y-axis to match image coordinates
ax.legend()
fig.show()


# %%
# Align source to target using homography
# --------------------------------------
# Extract vertices

src_points = np.array(source_polygon.coords.xy).T  # Shape (N, 2)
dst_points = np.array(target_polygon.coords.xy).T  # Shape (N, 2)


H = compute_homography_transform(src_points, dst_points)

print("Computed homography matrix H:")
print(np.array2string(H, precision=6, suppress_small=True))


# %%
# Apply homography transform to source polygon
# ------------------------------------------

transformed_polygon = source_polygon.apply_homography(H)

# %%
# Plot ROIs
fig, ax = plt.subplots()
target_polygon.plot(ax=ax, facecolor="m", alpha=0.5)
transformed_polygon.plot(ax=ax, facecolor="c", alpha=0.5)
ax.invert_yaxis()  # Flip the y-axis to match image coordinates
ax.legend()
fig.show()


# %%
# Now we will apply the homography to the pose tracks
# ------------------------------------------------------------------

poses_path = data_dir / "SLEAP_two-mice_octagon.analysis.h5"

ds = load_poses.from_file(poses_path, source_software="SLEAP", fps=60)

dims = ds.position.dims
print(f"Original dimensions: {dims}")
print(f"Original shape: {ds.position.shape}")

# Reorder the dimensions so 'space' is the LAST dimension.
# This ensures the final shape is (N, 2) after stacking.
position_reordered = ds.position.transpose(
    "time", "keypoints", "individuals", "space"
)
print(f"Reordered dimensions: {position_reordered.shape}")
print(f"Reordered shape: {position_reordered.shape}")

# Stack the first three dimensions into a new dimension called 'all_points'.
# 'space' is left unstacked at the end.
stacked_points = position_reordered.stack(
    all_points=["time", "keypoints", "individuals"]
)

print(f"Stacked dimensions: {stacked_points.dims}")
print(f"Stacked shape: {stacked_points.shape}")

# %%
# Apply homography transformation, input/output shapes are (N, 2)
stacked_points.loc[:, :] = transform_points_homography(
    stacked_points.values.T, H
).T

# Unstack to recover the original dimensions in their original order.
transformed_position = stacked_points.unstack("all_points")

# We need one final transpose to restore the *original* requested order.
transformed_position_final = transformed_position.transpose(
    "time", "space", "keypoints", "individuals"
)

print(f"\nFinal Transformed dimensions: {transformed_position_final.dims}")
print(f"Final Transformed shape: {transformed_position_final.shape}")

# %%
# Create a new dataset with transformed positions
ds_transformed = ds.copy()
ds_transformed["position"] = transformed_position_final

# %%
# Save transformed poses
# ----------------------------
output_path = data_dir / "SLEAP_two-mice_octagon_color3.analysis.h5"
save_poses.to_sleap_analysis_file(ds_transformed, output_path)

# %%
