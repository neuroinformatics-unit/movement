"""Load COCO keypoint annotations
==================================

Load a COCO keypoint annotation JSON file into ``movement``
and explore the resulting dataset.
"""

# %%
# Imports
# -------

import json
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from movement.io import load_poses

# %%
# About the COCO keypoint format
# --------------------------------
# The `COCO (Common Objects in Context)
# <https://cocodataset.org/#format-data>`_ dataset format is one
# of the most widely used standards for human pose estimation.
# COCO keypoint annotation files are JSON files containing three
# main sections:
#
# - **images**: a list of image entries (each with an ``id``).
# - **annotations**: a list of keypoint annotations, each
#   associated with an ``image_id`` and a ``category_id``.
#   The ``keypoints`` field is a flat array of
#   ``[x1, y1, v1, x2, y2, v2, ...]`` where ``v`` is the
#   visibility flag (0 = not labelled, 1 = labelled but
#   occluded, 2 = labelled and visible).
# - **categories**: defines the keypoint names and skeleton
#   connectivity.
#
# ``movement`` maps each image to a time frame, each annotation
# per image to an individual, and the visibility flags to
# confidence scores.

# %%
# Create a sample COCO file
# --------------------------
# For this example, we'll create a minimal COCO keypoint file
# with 10 frames, 2 individuals, and 5 keypoints per person.

keypoint_names = [
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
]
n_frames = 10
n_individuals = 2
n_keypoints = len(keypoint_names)

rng = np.random.default_rng(42)

coco_data = {
    "images": [
        {"id": i, "file_name": f"frame_{i:04d}.jpg"} for i in range(n_frames)
    ],
    "annotations": [],
    "categories": [
        {
            "id": 1,
            "name": "person",
            "keypoints": keypoint_names,
        }
    ],
}

ann_id = 0
for frame_idx in range(n_frames):
    for ind in range(n_individuals):
        # Simulate walking trajectories
        base_x = 100 + ind * 200 + frame_idx * 10
        base_y = 300 + ind * 50 - frame_idx * 5
        kps = []
        for k in range(n_keypoints):
            x = base_x + rng.normal(0, 3)
            y = base_y + k * 30 + rng.normal(0, 3)
            v = 2  # visible
            kps.extend([x, y, v])
        coco_data["annotations"].append(
            {
                "id": ann_id,
                "image_id": frame_idx,
                "category_id": 1,
                "keypoints": kps,
                "score": 0.85 + rng.uniform(0, 0.15),
                "track_id": ind,
            }
        )
        ann_id += 1

# Save to a temporary file
coco_path = Path(tempfile.mktemp(suffix=".json"))
with open(coco_path, "w") as f:
    json.dump(coco_data, f)
print(f"Created sample COCO file: {coco_path}")

# %%
# Load the COCO file into movement
# ---------------------------------
# Use :func:`movement.io.load_poses.from_coco_file` to load
# the COCO keypoint annotations. You can optionally specify
# an ``fps`` value to convert frame indices to seconds.

ds = load_poses.from_coco_file(coco_path, fps=30)
print(ds)

# %%
# Explore the dataset
# --------------------
# The resulting ``movement`` dataset has the familiar
# structure: ``position`` and ``confidence`` data variables
# with dimensions ``(time, space, keypoints, individuals)``.

print("Shape:", ds.position.shape)
print("Keypoints:", ds.coords["keypoints"].values)
print("Individuals:", ds.coords["individuals"].values)

# %%
# Visualise the trajectories
# ---------------------------
# Let's plot the nose trajectory for both individuals.

fig, ax = plt.subplots(figsize=(8, 5))
for ind in ds.coords["individuals"].values:
    x = ds.position.sel(keypoints="nose", individuals=ind, space="x")
    y = ds.position.sel(keypoints="nose", individuals=ind, space="y")
    ax.plot(x, y, "o-", label=ind, markersize=4)
ax.set_xlabel("x (pixels)")
ax.set_ylabel("y (pixels)")
ax.set_title("Nose trajectories from COCO data")
ax.legend()
ax.invert_yaxis()  # image coordinates
plt.tight_layout()
plt.show()

# %%
# Confidence scores
# ------------------
# The COCO visibility flags are mapped to confidence values:
# ``v=0`` → NaN position + 0 confidence,
# ``v=1`` → 0.5 × score,
# ``v=2`` → 1.0 × score.

print("Mean confidence:", float(ds.confidence.mean()))
print(
    "Min confidence:",
    float(ds.confidence.min()),
)

# %%
# Clean up
# --------

coco_path.unlink()
