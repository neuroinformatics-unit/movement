"""Visualization script for tracking data using Napari.

This script loads a CSV file containing object tracking data, extracts
bounding box information, and overlays the tracking points onto a video
in Napari.
"""

import ast  # To parse JSON-like strings

import napari
import numpy as np
import pandas as pd
import skimage.io

# Load the CSV tracking data
csv_file = "check_ffupsampled_dataset.csv"  # Update with actual path
df = pd.read_csv(csv_file)

# Extract frame numbers from filename
df["frame"] = df["filename"].apply(
    lambda x: int(x.split("/")[-1].split(".")[0])
)

# Parse bounding box JSON (convert string to dictionary)
df["bbox"] = df["region_shape_attributes"].apply(ast.literal_eval)

# Extract bounding box values
df["x_min"] = df["bbox"].apply(lambda b: b["x"])  # Top-left x
df["y_min"] = df["bbox"].apply(lambda b: b["y"])  # Top-left y
df["width"] = df["bbox"].apply(lambda b: b["width"])
df["height"] = df["bbox"].apply(lambda b: b["height"])

# Compute correct centroid (center of bounding box)
df["x_center"] = df["x_min"] + (df["width"] / 2)
df["y_center"] = df["y_min"] + (df["height"] / 2)

# Prepare centroid points for Napari (time, y, x)
points = np.column_stack((df["frame"], df["y_center"], df["x_center"]))

# Load the video
video_path = "MOCA_crab_video.mp4"  # Update with actual path
video = skimage.io.imread(video_path)

# Start Napari viewer
viewer = napari.Viewer()

# Add the video as an image layer
viewer.add_image(video, name="Crab Video")

# Add correctly computed centroid points
viewer.add_points(
    points, name="Centroid Tracking", size=5, face_color="red", opacity=0.8
)

# Run Napari
napari.run()
