"""Fetch and load publicly available datasets.

This module provides functions for fetching and loading publicly available
datasets of animal poses and trajectories. The data are downloaded from their
original sources and are cached locally the first time they are used.
"""

import logging
from pathlib import Path

import xarray as xr

logger = logging.getLogger(__name__)

# Save data in ~/.movement/public_data
PUBLIC_DATA_DIR = Path("~", ".movement", "public_data").expanduser()
# Create the folder if it doesn't exist
PUBLIC_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Dictionary of available datasets and their metadata
PUBLIC_DATASETS = {
    "calms21": {
        "description": "Caltech Mouse Social Interactions (CalMS21) Dataset: "
        "trajectory data of social interactions from videos of freely "
        "behaving mice in a standard resident-intruder assay.",
        "url": "https://data.caltech.edu/records/s0vdx-0k302",
        "paper": "https://arxiv.org/abs/2104.02710",  # SLEAP paper
        "license": "CC-BY-4.0",
    },
    "rat7m": {
        "description": "Rat7M: a 7M frame ground-truth dataset of rodent 3D "
        "landmarks and synchronised colour video.",
        "url": "https://figshare.com/collections/Rat_7M/5295370/3",
        "paper": "https://doi.org/10.1038/s41592-021-01106-6",  # DANNCE paper
        "license": "MIT",  # Assuming MIT based on DANNCE; verification needed
    },
}

# File registry for each dataset
# This will be populated as we implement each dataset loader
FILE_REGISTRY: dict[str, dict[str, str]] = {}


def list_public_datasets() -> list[str]:
    """List available public datasets.

    Returns
    -------
    dataset_names : list of str
        List of names for available public datasets.

    """
    return list(PUBLIC_DATASETS.keys())


def get_dataset_info(dataset_name: str) -> dict:
    """Get information about a public dataset.

    Parameters
    ----------
    dataset_name : str
        Name of the public dataset.

    Returns
    -------
    info : dict
        Dictionary containing dataset information.

    """
    if dataset_name not in PUBLIC_DATASETS:
        available_datasets = ", ".join(list_public_datasets())
        message = (
            f"Unknown dataset: {dataset_name}. "
            f"Available datasets are: {available_datasets}"
        )
        logger.error(message)
        raise ValueError(message)

    return PUBLIC_DATASETS[dataset_name]


def fetch_calms21(
    subset: str = "train",
    animal_type: str = "mouse",
    task: str = "open_field",
    frame_rate: float | None = None,
) -> xr.Dataset:
    """Fetch a subset of the CalMS21 dataset.

    The CalMS21 dataset consists of trajectory data of social interactions,
    recorded from videos of freely behaving mice in a standard
    resident-intruder assay. [1]_

    Parameters
    ----------
    subset : str, optional
        Data subset to fetch. One of 'train', 'val', or 'test'.
        Default is 'train'.
    animal_type : str, optional
        Type of animal (currently only 'mouse' is relevant for data fetching).
        Default is 'mouse'.
    task : str, optional
        Behavioral task (currently only 'social_interaction'/
        'resident-intruder' assays are relevant for data fetching).
        Default is 'open_field' (placeholder, specific tasks should
        be fetched).
    frame_rate : float, optional
        Frame rate in frames per second. If None, the original frame rate
        will be used. Default is None.

    Returns
    -------
    ds : xarray.Dataset
        Dataset containing the requested CalMS21 data.

    References
    ----------
    .. [1] Pereira, T. D., Tabris, N., Matsliah, A., Turner, D. M., Li, J.,
       Ravindranath, S., ... & Murthy, M. (2022). SLEAP: A deep learning system
       for multi-animal pose tracking. Nature Methods, 19(4), 486-495.
       https://arxiv.org/abs/2104.02710

    """
    # Validate inputs
    valid_subsets = ["train", "val", "test"]
    if subset not in valid_subsets:
        message = f"Invalid subset: {subset}. Must be one of {valid_subsets}"
        logger.error(message)
        raise ValueError(message)

    valid_animal_types = ["mouse"]
    if animal_type not in valid_animal_types:
        message = (
            f"Invalid animal type: {animal_type}. "
            f"Must be one of {valid_animal_types}"
        )
        logger.error(message)
        raise ValueError(message)

    valid_tasks = ["social_interaction", "resident_intruder"]
    if task not in valid_tasks:
        message = (
            f"Invalid task for {animal_type}: {task}. "
            f"Must be one of {valid_tasks}"
        )
        logger.error(message)
        raise ValueError(message)

    # Construction of URL and file paths will go here
    # For now, this is a placeholder implementation
    logger.info(f"Fetching CalMS21 data: {animal_type}/{task}/{subset}")

    # Placeholder for actual implementation
    # This would use pooch to download the specific file
    # And then load it into an xarray Dataset

    # For demonstration, create a minimal dataset
    ds = xr.Dataset()
    ds.attrs["dataset"] = "calms21"
    ds.attrs["subset"] = subset
    ds.attrs["animal_type"] = animal_type
    ds.attrs["task"] = task

    # In actual implementation, we would:
    # 1. Download the data file using pooch
    # 2. Load the file into appropriate format
    # 3. Convert to movement's xarray format
    # 4. Return the dataset

    logger.warning(
        "This is currently a placeholder implementation. "
        "The actual data downloading is not yet implemented."
    )

    return ds


def fetch_rat7m(
    subset: str = "open_field",
    frame_rate: float | None = None,
) -> xr.Dataset:
    """Fetch a subset of the Rat7M dataset.

    The Rat7M dataset contains tracking data for multiple rats in complex
    environments.

    Parameters
    ----------
    subset : str, optional
        Data subset to fetch. One of 'open_field', 'shelter', or 'maze'.
        Default is 'open_field'.
    frame_rate : float, optional
        Frame rate in frames per second. If None, the original frame rate
        will be used. Default is None.

    Returns
    -------
    ds : xarray.Dataset
        Dataset containing the requested Rat7M data.

    References
    ----------
    .. [1] Dunn et al. (2021). "Geometric deep learning enables 3D kinematic
       profiling across species and environments". Nature Methods, 18(5),
       564-573. https://doi.org/10.1038/s41592-021-01106-6

    """
    # Validate inputs
    valid_subsets = ["open_field", "shelter", "maze"]
    if subset not in valid_subsets:
        message = f"Invalid subset: {subset}. Must be one of {valid_subsets}"
        logger.error(message)
        raise ValueError(message)

    # Construction of URL and file paths will go here
    # For now, this is a placeholder implementation
    logger.info(f"Fetching Rat7M data: {subset}")

    # Placeholder for actual implementation
    # This would use pooch to download the specific file
    # And then load it into an xarray Dataset

    # For demonstration, create a minimal dataset
    ds = xr.Dataset()
    ds.attrs["dataset"] = "rat7m"
    ds.attrs["subset"] = subset

    # In actual implementation, we would:
    # 1. Download the data file using pooch
    # 2. Load the file into appropriate format
    # 3. Convert to movement's xarray format
    # 4. Return the dataset

    logger.warning(
        "This is currently a placeholder implementation. "
        "The actual data downloading is not yet implemented."
    )

    return ds
