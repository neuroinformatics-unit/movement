"""Test suite for the public_data module."""

import pytest
from unittest.mock import patch

from movement import public_data


def test_list_public_datasets():
    """Test listing available public datasets."""
    datasets = public_data.list_public_datasets()
    assert isinstance(datasets, list)
    assert len(datasets) > 0
    assert "calms21" in datasets
    assert "rat7m" in datasets


def test_get_dataset_info():
    """Test getting information about a public dataset."""
    # Test valid dataset
    info = public_data.get_dataset_info("calms21")
    assert isinstance(info, dict)
    assert "description" in info
    assert "url" in info
    assert "paper" in info
    assert "license" in info
    
    # Test invalid dataset
    with pytest.raises(ValueError, match="Unknown dataset"):
        public_data.get_dataset_info("nonexistent_dataset")


@pytest.mark.parametrize(
    "subset, animal_type, task",
    [
        ("train", "mouse", "open_field"),
        ("val", "fly", "courtship"),
        ("test", "ciona", "social_investigation"),
    ],
)
def test_fetch_calms21_valid_inputs(subset, animal_type, task):
    """Test fetching CalMS21 dataset with valid inputs."""
    with patch("movement.public_data.logger.warning"):  # Suppress warning
        ds = public_data.fetch_calms21(
            subset=subset, animal_type=animal_type, task=task
        )
        assert "dataset" in ds.attrs
        assert ds.attrs["dataset"] == "calms21"
        assert ds.attrs["subset"] == subset
        assert ds.attrs["animal_type"] == animal_type
        assert ds.attrs["task"] == task


@pytest.mark.parametrize(
    "subset, animal_type, task, error_match",
    [
        ("invalid", "mouse", "open_field", "Invalid subset"),
        ("train", "invalid", "open_field", "Invalid animal type"),
        ("train", "mouse", "invalid", "Invalid task for mouse"),
        # Cross-species task mismatch
        ("train", "fly", "open_field", "Invalid task for fly"),
    ],
)
def test_fetch_calms21_invalid_inputs(subset, animal_type, task, error_match):
    """Test fetching CalMS21 dataset with invalid inputs."""
    with pytest.raises(ValueError, match=error_match):
        public_data.fetch_calms21(
            subset=subset, animal_type=animal_type, task=task
        )


@pytest.mark.parametrize(
    "subset",
    ["open_field", "shelter", "maze"],
)
def test_fetch_rat7m_valid_inputs(subset):
    """Test fetching Rat7M dataset with valid inputs."""
    with patch("movement.public_data.logger.warning"):  # Suppress warning
        ds = public_data.fetch_rat7m(subset=subset)
        assert "dataset" in ds.attrs
        assert ds.attrs["dataset"] == "rat7m"
        assert ds.attrs["subset"] == subset


def test_fetch_rat7m_invalid_inputs():
    """Test fetching Rat7M dataset with invalid inputs."""
    with pytest.raises(ValueError, match="Invalid subset"):
        public_data.fetch_rat7m(subset="invalid") 