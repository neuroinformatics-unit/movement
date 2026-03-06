import json
from pathlib import Path
from movement.io.load_poses import from_mmpose_file

# Create a sample MMPose file
sample_data = [
    {
        "frame_id": 0,
        "instances": [
            {
                "keypoints": [[100.0, 200.0], [150.0, 250.0]],
                "keypoint_scores": [0.95, 0.88],
                "bbox": [50, 50, 200, 300],
                "bbox_score": 0.92,
                "track_id": 1
            }
        ]
    },
    {
        "frame_id": 1,
        "instances": [
            {
                "keypoints": [[102.0, 205.0], [155.0, 255.0]],
                "keypoint_scores": [0.94, 0.89],
                "bbox": [52, 55, 205, 305],
                "bbox_score": 0.91,
                "track_id": 1
            }
        ]
    }
]

file_path = Path("mmpose_example.json")
with open(file_path, "w") as f:
    json.dump(sample_data, f)

print(f"Created sample MMPose file: {file_path}")

# Load the data
print("Loading MMPose file...")
ds = from_mmpose_file(file_path, fps=30)

# Display the dataset
print("\nMovement Dataset Representation:")
print(ds)

print("\nPosition data (first two frames):")
print(ds.position.values[:2])

# Clean up
file_path.unlink()
print(f"\nCleaned up sample file: {file_path}")
