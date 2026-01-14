#!/usr/bin/env python3
"""Test poses_to_bboxes with real SLEAP dataset."""

from movement import sample_data
from movement.io import load_poses
from movement.transforms import poses_to_bboxes

# Load multi-animal dataset
file_path = sample_data.fetch_dataset_paths(
    "SLEAP_three-mice_Aeon_proofread.analysis.h5"
)["poses"]
ds = load_poses.from_sleap_file(file_path, fps=50)

print("=== INPUT POSES ===")
print(f"Dimensions: {dict(ds.dims)}")
print(f"Individuals: {list(ds.coords['individuals'].values)}")
print(f"Keypoints: {list(ds.coords['keypoints'].values)}")
print(f"Data variables: {list(ds.data_vars)}")

# Convert to bboxes
bboxes = poses_to_bboxes(ds, padding_px=10)

print("\n=== OUTPUT BBOXES ===")
print(f"Dimensions: {dict(bboxes.dims)}")
print(f"Individuals: {list(bboxes.coords['individuals'].values)}")
print(f"Data variables: {list(bboxes.data_vars)}")
print(f"ds_type attribute: {bboxes.attrs.get('ds_type', 'MISSING')}")

# Verify structure matches movement bboxes format
assert "position" in bboxes.data_vars, "Missing 'position' (centroids)"
assert "shape" in bboxes.data_vars, "Missing 'shape' (width/height)"
assert set(bboxes.coords["space"].values) == {"x", "y"}, "Space should be 2D"
assert "keypoints" not in bboxes.dims, "Bboxes should not have keypoints dim"

# Verify individual IDs preserved
assert list(bboxes.coords["individuals"].values) == list(ds.coords["individuals"].values), "Individual IDs not preserved"

# Verify time dimension preserved
assert bboxes.dims["time"] == ds.dims["time"], "Time dimension changed"

# Verify metadata preserved
print(f"\n=== METADATA ===")
print(f"fps: {bboxes.attrs.get('fps', 'MISSING')}")
print(f"source_software: {bboxes.attrs.get('source_software', 'MISSING')}")
print(f"source_file: {bboxes.attrs.get('source_file', 'MISSING')}")

print("\n✅ All structural checks passed!")
