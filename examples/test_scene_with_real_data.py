"""
Test: Create Scene using real pose data
========================================

This script loads sample pose data using `load_poses`, constructs Individual objects,
and tests the Scene logic for handling heterogeneous keypoints.
"""

from movement import sample_data
from movement.io import load_poses
from movement.data.heterogeneous_keypoints import Individual, Scene
import numpy as np

# Load sample dataset (SLEAP - 3 mice)
file_path = sample_data.fetch_dataset_paths(
    "SLEAP_three-mice_Aeon_proofread.analysis.h5"
)["poses"]

# Load the dataset
ds = load_poses.from_sleap_file(file_path, fps=50)
position = ds.position

# Create Scene
scene = Scene()

# Loop through all individuals
for ind_name in position.individuals.values:
    kp_names = list(position.keypoints.values)
    data = position.sel(individuals=ind_name).values  # shape: (frames, space, keypoints)
    
    # Transpose to match our format: (frames, keypoints, space)
    data = np.transpose(data, (0, 2, 1))

    # Create Individual
    ind = Individual(ind_name, data, kp_names)
    scene.add_individual(ind)

# Print results
print("Individuals:", list(scene.individuals.keys()))
print("Common keypoints:", scene.get_common_keypoints())
