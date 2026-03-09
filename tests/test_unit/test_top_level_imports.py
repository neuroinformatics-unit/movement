"""Tests for top-level imports from the movement package."""


def test_load_dataset_importable_from_top_level():
    """Test that load_dataset can be imported from movement directly."""
    from movement import load_dataset

    assert callable(load_dataset)


def test_load_multiview_dataset_importable_from_top_level():
    """Test that load_multiview_dataset can be imported from movement."""
    from movement import load_multiview_dataset

    assert callable(load_multiview_dataset)


def test_top_level_load_dataset_is_same_as_io():
    """Top-level load_dataset is the same object as movement.io.load's."""
    from movement import load_dataset
    from movement.io.load import load_dataset as io_load_dataset

    assert load_dataset is io_load_dataset


def test_top_level_load_multiview_dataset_is_same_as_io():
    """Top-level load_multiview_dataset is the same as movement.io.load's."""
    from movement import load_multiview_dataset
    from movement.io.load import (
        load_multiview_dataset as io_load_multiview_dataset,
    )

    assert load_multiview_dataset is io_load_multiview_dataset
