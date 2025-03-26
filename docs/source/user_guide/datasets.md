# Working with Public Datasets

In addition to sample data for testing and examples, `movement` provides access to publicly available datasets of animal poses and trajectories. These datasets can be useful for research, method development, benchmarking, and learning.

## Available Datasets

You can list the available public datasets using:

```python
from movement import list_public_datasets

datasets = list_public_datasets()
print(datasets)
```

To get more information about a specific dataset:

```python
from movement import get_dataset_info

info = get_dataset_info("calms21")
print(info["description"])
print(info["url"])
print(info["paper"])
print(info["license"])
```

## CalMS21 Dataset

The [CalMS21 dataset](https://data.caltech.edu/records/g6fjs-ceqwp) contains multi-animal pose tracking data for various animal types and behavioral tasks.

```python
from movement import public_data

# Fetch mouse data from the open field task
mouse_data = public_data.fetch_calms21(
    subset="train",
    animal_type="mouse",
    task="open_field",
)

# Fetch fly data from the courtship task
fly_data = public_data.fetch_calms21(
    subset="train",
    animal_type="fly",
    task="courtship",
)
```

The available parameters are:

- `subset`: "train", "val", or "test"
- `animal_type`: "mouse", "fly", or "ciona"
- `task`: Depends on the animal type
  - For mouse: "open_field", "social_interaction", "resident_intruder"
  - For fly: "courtship", "egg_laying", "aggression"
  - For ciona: "social_investigation"
- `frame_rate`: Optional, to override the original frame rate

## Rat7M Dataset

The [Rat7M dataset](https://data.caltech.edu/records/bpkf7-jae29) contains tracking data for multiple rats in complex environments.

```python
from movement import public_data

# Fetch data from the open field task
rat_data = public_data.fetch_rat7m(subset="open_field")
```

The available parameters are:

- `subset`: "open_field", "shelter", or "maze"
- `frame_rate`: Optional, to override the original frame rate

## Data Caching

Downloaded datasets are cached locally in the `~/.movement/public_data` directory. This means that after the first download, subsequent requests for the same dataset will be faster as they'll use the local copy.

## Working with the Data

Once loaded, these datasets are returned as standard `movement` xarray Datasets, allowing you to apply all the analysis and visualization tools available in the package:

```python
from movement import public_data
import matplotlib.pyplot as plt

# Fetch data
ds = public_data.fetch_calms21(animal_type="mouse", task="open_field")

# Access position data
position = ds.position

# Compute kinematics
from movement import kinematics
velocity = kinematics.compute_velocity(position)
speed = kinematics.compute_speed(position)

# Visualize
from movement.plots import plot_centroid_trajectory
fig, ax = plot_centroid_trajectory(position)
plt.show()
```

## Citation

When using these public datasets in your research, please cite the original papers:

- CalMS21: Sun et al. (2022). "Multi-animal pose estimation, identification and tracking with DeepLabCut". Nature Methods, 19(4), 496-504. https://doi.org/10.1038/s41592-022-01426-1

- Rat7M: Dunn et al. (2021). "Geometric deep learning enables 3D kinematic profiling across species and environments". Nature Methods, 18(5), 564-573. https://doi.org/10.1038/s41592-021-01106-6
