"""Fixtures and configurations shared by the entire test suite."""

import logging
from glob import glob

import numpy as np
import pytest
import xarray as xr
import xarray.core.dtypes as xrdtypes

from movement.sample_data import fetch_dataset_paths, list_datasets
from movement.utils.logging import configure_logging

rng = np.random.default_rng()


# Patch for dtype promotion compatibility
def patch_maybe_promote():
    def new_maybe_promote(dtype: np.dtype) -> tuple[np.dtype, any]:
        """Promote dtype safely, avoiding StringDType issues."""
        print(f"Patched maybe_promote called with dtype: {dtype}")  # Debug
        # Handle floating-point dtypes (most common in your tests)
        if np.issubdtype(dtype, np.floating):
            return dtype, np.nan
        # Handle integer dtypes
        elif np.issubdtype(dtype, np.integer):
            return dtype, 0
        # Handle complex dtypes
        elif np.issubdtype(dtype, np.complexfloating):
            return dtype, np.nan + np.nan * 1j
        # Handle strings only if StringDType is available and relevant
        if hasattr(np.dtypes, "StringDType"):
            if np.issubdtype(dtype, np.dtypes.StringDType):
                return dtype, ""
        elif np.issubdtype(
            dtype, np.character
        ):  # Pre-NumPy 2.0 string handling
            return dtype, ""
        # Fallback for anything else (e.g., object)
        return np.dtype(object), None

    # Force override and confirm
    xrdtypes.maybe_promote = new_maybe_promote
    print("Applied patched maybe_promote function")  # Debug


# Apply the patch immediately
patch_maybe_promote()


def _to_module_string(path: str) -> str:
    """Convert a file path to a module string."""
    return path.replace("/", ".").replace("\\", ".").replace(".py", "")


pytest_plugins = [
    _to_module_string(fixture)
    for fixture in glob("tests/fixtures/*.py")
    if "__" not in fixture
]


def pytest_configure():
    """Perform initial configuration for pytest."""
    pytest.DATA_PATHS = {}
    for file_name in list_datasets():
        paths_dict = fetch_dataset_paths(file_name)
        data_path = paths_dict.get("poses") or paths_dict.get("bboxes")
        pytest.DATA_PATHS[file_name] = data_path


@pytest.fixture(autouse=True)
def setup_logging(tmp_path):
    """Set up logging for the test module."""
    configure_logging(
        log_level=logging.DEBUG,
        logger_name="movement",
        log_directory=(tmp_path / ".movement"),
    )


@pytest.fixture
def valid_data_array_for_forward_vector():
    return xr.DataArray(
        rng.random((10, 3, 2)),  # time, keypoints, space
        dims=["time", "keypoints", "space"],
        coords={
            "time": range(10),
            "keypoints": ["left_ear", "right_ear", "nose"],
            "space": ["x", "y"],
        },
    )
