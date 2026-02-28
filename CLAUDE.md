# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

movement is a Python toolbox for analysing animal body movements across space and time. It provides a standardised, modular interface for processing motion tracking data from tools like DeepLabCut, SLEAP, LightningPose, Anipose, and VIA-tracks. The package handles data loading, cleaning, transformation, kinematic analysis, and visualization.

## Development Commands

### Environment Setup
```bash
# Create virtual environment with uv
uv venv --python 3.13
source .venv/bin/activate  # On macOS and Linux
.venv\Scripts\activate     # On Windows PowerShell

# Install in editable mode with dev dependencies
uv pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Testing
```bash
# Run all tests with coverage
pytest

# Run specific test file
pytest tests/test_unit/test_io/test_load_poses.py

# Run specific test function
pytest tests/test_unit/test_io/test_load_poses.py::test_from_dlc_file

# Run tests matching pattern
pytest -k "test_load"

# Run with verbose output
pytest -v

# Run without coverage (faster)
pytest --no-cov
```

### Linting and Formatting
```bash
# Run pre-commit hooks on staged files
pre-commit run

# Run pre-commit hooks on all files
pre-commit run -a

# Auto-fix with ruff
ruff check --fix

# Type checking with mypy
mypy movement
```

### Documentation
```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build documentation (from docs/ directory)
cd docs
make html

# Clean and rebuild documentation
make clean html

# Check external links
make linkcheck

# View built documentation
open build/html/index.html  # macOS
```

## High-Level Architecture

### Core Data Structure: xarray Datasets

The entire package is built around **xarray Datasets** as the central data structure. Understanding this is crucial for working with the codebase.

**Poses Dataset:**
- Dimensions: `(time, space, keypoints, individuals)`
- Variables:
  - `position`: shape `(n_frames, n_space, n_keypoints, n_individuals)`
  - `confidence`: shape `(n_frames, n_keypoints, n_individuals)`
- Coordinates:
  - `time`: seconds (if fps provided) or frame numbers
  - `space`: `["x", "y"]` or `["x", "y", "z"]`
  - `keypoints`: list of keypoint names
  - `individuals`: list of individual names
- Attributes: `source_software`, `fps`, `time_unit`, `ds_type="poses"`, `log`

**Bboxes Dataset:**
- Dimensions: `(time, space, individuals)`
- Variables:
  - `position`: centroid positions
  - `shape`: bounding box dimensions
  - `confidence`: detection confidence

### Module Organization

**`io/`** - Data loading and saving
- `load.py`: Unified entry points `load_dataset()` and `load_multiview_dataset()` with a `@register_loader` decorator pattern for format-specific loaders
- `load_poses.py`: Format-specific pose loaders (`from_dlc_file()`, `from_sleap_file()`, `from_lp_file()`, `from_anipose_file()`). Note: `from_file()` and `from_multiview_files()` are deprecated in favour of `load_dataset()` / `load_multiview_dataset()`
- `load_bboxes.py`: Format-specific bbox loaders, including VIA-tracks support. Note: `from_file()` is deprecated in favour of `load_dataset()`
- `save_poses.py`, `save_bboxes.py`: Export to various formats
- `nwb.py`: NWB (Neurodata Without Borders) format support

**`validators/`** - Multi-layer validation system
- `files.py`: File format validation using attrs classes (`ValidFile`, `ValidHDF5`, `ValidDeepLabCutCSV`, etc.)
- `datasets.py`: Dataset validation (`ValidPosesDataset`, `ValidBboxesDataset`) with automatic defaults and shape checking
- `arrays.py`: Array validation (`validate_dims_coords()`)

**`filtering.py`** - Signal processing
- `filter_by_confidence()`: Mask low-confidence points
- `interpolate_over_time()`: Fill NaN gaps (linear, nearest, cubic)
- `savgol_filter()`, `rolling_filter()`: Smooth trajectories

**`transforms.py`** - Spatial transformations
- `scale()`: Convert pixel to physical units
- `compute_homography_transform()`: Perspective correction

**`kinematics/`** - Motion analysis
- `kinematics.py`: Core kinematics (velocity, acceleration, displacement)
- `distances.py`: Pairwise distances between keypoints
- `orientation.py`: Forward vectors, head direction
- `kinetic_energy.py`: Energy computations

**`roi/`** - Region of Interest analysis
- `base.py`: Abstract base class for ROIs
- `polygon.py`: Polygon regions (occupancy, crossing events)
- `line.py`: Line regions (crossing detection)
- `conditions.py`: Boolean conditions on ROIs

**`napari/`** - GUI plugin
- `meta_widget.py`: `MovementMetaWidget` - collapsible container wrapping all napari widgets
- `loader_widgets.py`: File loading widgets
- `convert.py`: xarray → napari layer conversion
- `layer_styles.py`: Visual styling for layers
- `regions_widget.py`: Interactive region drawing and analysis

**`plots/`** - Visualization
- `trajectory.py`: Plot trajectories over time
- `occupancy.py`: Heatmaps of spatial occupancy

**`utils/`** - Cross-cutting utilities
- `logging.py`: `MovementLogger` wrapper around loguru with `log_to_attrs()` decorator for operation provenance
- `broadcasting.py`: `make_broadcastable()` decorator for applying 1D functions across dimensions
- `vector.py`: Vector operations
- `reports.py`: Generate analysis reports

**`sample_data.py`** - Sample dataset management via pooch

**`cli_entrypoint.py`** - CLI commands (`movement info` for diagnostics, `movement launch` for napari)

### Data Flow Pipeline

```
1. FILE INPUT (io/load_*.py)
   └─> File format validation (validators/files.py)

2. DATA VALIDATION (validators/datasets.py)
   └─> Extract arrays, validate shapes, set defaults

3. DATASET CREATION (_ds_from_valid_data)
   └─> Create xr.Dataset with proper dims/coords/attrs

4. DATA CLEANING (filtering.py)
   └─> filter_by_confidence → interpolate_over_time → smooth

5. TRANSFORMATIONS (transforms.py)
   └─> scale, homography transforms

6. KINEMATIC ANALYSIS (kinematics/)
   └─> velocity, acceleration, distances, orientation

7. ROI ANALYSIS (roi/)
   └─> Define regions, compute occupancy, detect events

8. OUTPUT
   └─> Save (io/save_*.py), Visualize (plots/), GUI (napari/)
```

### Key Architectural Patterns

**1. Validation with attrs:**
All validators use attrs `@define` classes with field validators. Example:
```python
@define(kw_only=True)
class ValidPosesDataset:
    position_array: np.ndarray = field()
    # validators ensure data integrity
    # __attrs_post_init__ sets defaults
```

**2. Operation Logging with Decorators:**
The `@log_to_attrs` decorator automatically logs all operations to `dataset.attrs["log"]` as JSON, creating an audit trail:
```python
@log_to_attrs
def filter_by_confidence(data, threshold):
    # Operation automatically logged
    return filtered_data
```

**3. Broadcasting Utilities:**
The `@make_broadcastable` decorator allows 1D functions to work across xarray dimensions automatically.

**4. Immutable Operations:**
All processing functions return new xarray objects rather than modifying in-place.

**5. Framework-Agnostic Interface:**
Common interface for multiple pose estimation tools through format-specific loaders that all produce the same xarray structure.

**6. Loader Registration:**
The `@register_loader` decorator in `io/load.py` registers format-specific loaders (from `load_poses.py`, `load_bboxes.py`, `nwb.py`) so that `load_dataset()` can dispatch by `source_software` name.

## Important Conventions

### Dimensions and Coordinates
- Always use `DIM_NAMES` from `movement.validators.arrays` for consistency
- Time coordinate should be in seconds if fps is available, otherwise frame numbers
- Space coordinate is `["x", "y"]` or `["x", "y", "z"]`

### Dataset Attributes
- `ds_type`: either `"poses"` or `"bboxes"`
- `source_software`: tracking software name
- `fps`: frames per second (can be None)
- `log`: JSON list of operations performed
- Additional attributes preserved during operations

### Logging
- Use `from movement.utils.logging import logger` (not standard logging)
- Log at appropriate levels: `debug()`, `info()`, `warning()`, `error()`, `exception()`
- Use `logger.error()` and `logger.exception()` to log and raise exceptions in one line

### Docstrings
- Follow numpydoc style for all public functions, classes, and methods
- Include Parameters, Returns, Raises, Notes, Examples sections as appropriate
- Docstrings auto-generate API reference documentation

### Testing
- Unit tests in `tests/test_unit/` mirror package structure
- Integration tests in `tests/test_integration/`
- Use fixtures from `tests/fixtures/` (check before adding new ones)
- Use sample data via `pytest.DATA_PATHS` dictionary (populated in conftest.py)
- Don't commit large data files; add to external GIN repository instead

### Code Style
- Line length: 79 characters (ruff enforces this)
- Use ruff for linting and formatting (configured in pyproject.toml)
- Type hints checked with mypy (some external modules ignored)
- Pre-commit hooks auto-fix many issues

## Working with Sample Data

Sample datasets are hosted on GIN (German Neuroinformatics Node) and managed via pooch:

```python
from movement import sample_data

# List available datasets
datasets = sample_data.list_datasets()

# Fetch and load a dataset
ds = sample_data.fetch_dataset("DLC_single-wasp.predictions.h5")

# Get file paths without loading
paths = sample_data.fetch_dataset_paths("DLC_single-wasp.predictions.h5")
# Returns: {"poses": "path/to/file", "frame": "...", "video": "..."}
```

## Release Process

Releases are managed via git tags and GitHub Actions:

```bash
# Create and push a version tag
git tag -a v1.0.0 -m "Bump to version 1.0.0"
git push --follow-tags
```

This triggers:
1. Linting checks
2. Tests across Python 3.11, 3.12, 3.13 on Linux, macOS, Windows
3. Build source distribution and wheels
4. Deployment to PyPI (automatic for tags)

Version numbers follow semantic versioning (MAJOR.MINOR.PATCH).

## napari Plugin

The napari plugin provides interactive visualization. Key components:

- Entry point defined in `pyproject.toml`: `entry-points."napari.manifest".movement`
- `napari.yaml`: Plugin manifest
- `meta_widget.py`: `MovementMetaWidget` - main collapsible container widget
- `loader_widgets.py`: File loading interfaces
- `convert.py`: Converts xarray Datasets to napari layers (Tracks, Points, Shapes)
- `regions_widget.py`: Interactive region drawing with Qt widgets

To test the plugin locally:
```bash
pip install -e ".[napari]"
napari
# Or use the CLI shortcut:
movement launch
```

## Common Pitfalls

1. **Don't modify datasets in-place**: Always return new xarray objects
2. **Check for existing fixtures**: Review `tests/fixtures/` before creating new test data
3. **Use xarray accessors carefully**: Some operations require specific dimensions
4. **Preserve attributes**: When creating new DataArrays/Datasets, copy relevant attributes
5. **Test with multiple formats**: Changes to I/O should be tested across DLC, SLEAP, etc.
6. **Update API docs**: If adding package modules that expose public APIs via `__init__.py`, add to `PACKAGE_MODULES` in `docs/make_api.py`
