"""Loading Human Pose Tracking Data
================================

This example demonstrates how to load human pose tracking data from various
popular formats: COCO, MMPose, and FreeMoCap.

The ``movement`` package provides a unified interface for loading these formats
into a standardized ``xarray.Dataset`` structure.
"""

import matplotlib.pyplot as plt
import numpy as np

# %%
# Loading COCO Keypoint Data
# --------------------------
# COCO (Common Objects in Context) is a widely used format for human pose
# estimation. ``movement`` can load COCO JSON files that contain keypoint
# annotations for one or more individuals.
#
# Since we don't have a built-in COCO sample dataset yet, we'll demonstrate
# how you would call the loader. In a real scenario, you would have a
# .json file from your pose estimation pipeline.

# coco_ds = load_dataset(
#     "path/to/coco_keypoints.json",
#     source_software="COCO",
#     fps=30
# )

# %%
# Loading MMPose Output
# ---------------------
# MMPose is an open-source toolbox for pose estimation based on PyTorch.
# It supports a wide range of models and outputs data in JSON format.
#
# Similar to COCO, you can load MMPose JSON files using:

# mmpose_ds = load_dataset(
#     "path/to/mmpose_output.json",
#     source_software="MMPose",
#     fps=30
# )

# %%
# Loading FreeMoCap Data
# ----------------------
# FreeMoCap is an open-source tool for markerless motion capture.
# It produces a directory containing 3D coordinates for body joints,
# which ``movement`` can aggregate into a single dataset.
#
# You can load FreeMoCap data from a session directory:

# freemocap_ds = load_dataset(
#     "path/to/freemocap_session/",
#     source_software="FreeMocap",
#     fps=30
# )


# %%
# Visualizing the Trajectories
# ----------------------------
# Once loaded, you can explore the data using standard ``movement``
# and ``xarray`` tools. For example, plotting the trajectory of a specific
# keypoint over time.

# Let's use some dummy data to illustrate the plotting
# (In a real example, you would use one of the loaded datasets above)

time = np.linspace(0, 10, 100)
x = np.sin(time)
y = np.cos(time)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(x, y, label="Trajectory (e.g., Nose)")
ax.set_xlabel("X coordinate")
ax.set_ylabel("Y coordinate")
ax.set_title("Sample Keypoint Trajectory")
ax.legend()
plt.show()

# %%
# Summary
# -------
# By supporting COCO, MMPose, and FreeMoCap, ``movement`` bridges the gap
# between common human pose estimation tools and a powerful Python-based
# analysis ecosystem.
