"""Tests for Kalman filtering implementation in movement tracking."""

import numpy as np
import pytest
import xarray as xr

from movement.kalman_filter import (
    apply_kalman_filter_xarray,
    load_sleap_data_xarray,
)
from movement.sample_data import fetch_dataset_paths


@pytest.fixture
def sample_dataset():
    """Create a sample dataset for Kalman filter testing."""
    frames = 20
    x_values = np.linspace(0, 10, frames) + np.random.normal(0, 0.5, frames)
    y_values = np.linspace(0, 5, frames) + np.random.normal(0, 0.5, frames)

    dataset = xr.Dataset(
        {
            "x": ("frame", x_values),
            "y": ("frame", y_values),
        },
        coords={"frame": np.arange(frames)},
    )
    return dataset


def test_kalman_filter_runs(sample_dataset):
    """Ensure Kalman filter runs without crashing.

    Produces expected output variables.
    """
    filtered_data = apply_kalman_filter_xarray(sample_dataset)

    assert "x_filtered" in filtered_data
    assert "y_filtered" in filtered_data
    assert "x_velocity" in filtered_data
    assert "y_velocity" in filtered_data
    assert "x_acceleration" in filtered_data
    assert "y_acceleration" in filtered_data


def test_kalman_filter_smooths_noise(sample_dataset):
    """Ensure Kalman filter reduces noise compared to raw data."""
    filtered_data = apply_kalman_filter_xarray(sample_dataset)

    original_noise_x = np.std(sample_dataset["x"].values)
    filtered_noise_x = np.std(filtered_data["x_filtered"].values)

    original_noise_y = np.std(sample_dataset["y"].values)
    filtered_noise_y = np.std(filtered_data["y_filtered"].values)

    assert filtered_noise_x < original_noise_x, (
        "Kalman filter should reduce noise in x"
    )
    assert filtered_noise_y < original_noise_y, (
        "Kalman filter should reduce noise in y"
    )


def test_kalman_filter_does_not_create_nans(sample_dataset):
    """Ensure Kalman filter does not introduce NaN values."""
    filtered_data = apply_kalman_filter_xarray(sample_dataset)

    assert not np.isnan(filtered_data["x_filtered"]).any()
    assert not np.isnan(filtered_data["y_filtered"]).any()
    assert not np.isnan(filtered_data["x_velocity"]).any()
    assert not np.isnan(filtered_data["y_velocity"]).any()
    assert not np.isnan(filtered_data["x_acceleration"]).any()
    assert not np.isnan(filtered_data["y_acceleration"]).any()


def test_kalman_filter_on_real_data():
    """Ensure Kalman filter works with an actual movement dataset."""
    file_path = fetch_dataset_paths(
        "SLEAP_three-mice_Aeon_proofread.analysis.h5"
    )["poses"]

    try:
        dataset = load_sleap_data_xarray(file_path)
    except FileNotFoundError:
        pytest.skip("Real dataset file not found, skipping test.")

    filtered_data = apply_kalman_filter_xarray(dataset)

    assert "x_filtered" in filtered_data
    assert "y_filtered" in filtered_data


def test_kalman_filter_corrects_outliers():
    """Ensure Kalman filter smooths extreme outliers."""
    frames = 30
    x_values = np.linspace(0, 15, frames)
    y_values = np.linspace(0, 8, frames)

    # Introduce outliers
    x_values[10] += 20  # Large sudden jump
    y_values[15] -= 15  # Large sudden drop

    dataset = xr.Dataset(
        {
            "x": ("frame", x_values),
            "y": ("frame", y_values),
        },
        coords={"frame": np.arange(frames)},
    )

    filtered_data = apply_kalman_filter_xarray(dataset)

    assert abs(filtered_data["x_filtered"].values[10] - x_values[10]) > 5, (
        "Kalman filter should correct large jumps"
    )

    assert abs(filtered_data["y_filtered"].values[15] - y_values[15]) > 5, (
        "Kalman filter should correct large drops"
    )


def test_file_not_found():
    """Ensure load_sleap_data_xarray raises FileNotFoundError.

    This should happen when an invalid file is provided.
    """
    with pytest.raises(FileNotFoundError):
        load_sleap_data_xarray("non_existent_file.h5")


def test_singular_matrix_case(mocker):
    """Ensure apply_kalman_filter_xarray handles singular matrix inversion.

    It should fail gracefully.
    """
    dataset = xr.Dataset(
        {
            "x": ("frame", np.linspace(0, 10, 20)),
            "y": ("frame", np.linspace(0, 5, 20)),
        },
        coords={"frame": np.arange(20)},
    )

    mocker.patch("numpy.linalg.pinv", side_effect=np.linalg.LinAlgError)

    filtered_data = apply_kalman_filter_xarray(dataset)

    assert "x_filtered" in filtered_data
    assert "y_filtered" in filtered_data
