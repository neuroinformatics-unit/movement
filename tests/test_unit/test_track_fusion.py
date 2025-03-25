"""Tests for track fusion functions."""

import numpy as np
import pytest
import xarray as xr

from movement.track_fusion import (
    align_datasets,
    fuse_tracks,
    fuse_tracks_kalman,
    fuse_tracks_mean,
    fuse_tracks_median,
    fuse_tracks_reliability,
    fuse_tracks_weighted,
)


@pytest.fixture
def mock_datasets():
    """Create mock datasets for testing track fusion."""
    # Create two simple datasets with different time points and some NaNs
    # Dataset 1: More reliable (fewer NaNs)
    time1 = np.arange(0, 10, 1)
    pos1 = np.zeros((10, 1, 2))
    # Simple straight line with a slope
    pos1[:, 0, 0] = np.arange(0, 10, 1)  # x coordinate
    pos1[:, 0, 1] = np.arange(0, 10, 1)  # y coordinate
    # Add some NaNs
    pos1[3, 0, :] = np.nan

    # Dataset 2: Less reliable (more NaNs)
    time2 = np.arange(0, 10, 1)
    pos2 = np.zeros((10, 1, 2))
    # Similar trajectory but with some noise
    pos2[:, 0, 0] = np.arange(0, 10, 1) + np.random.normal(0, 0.5, 10)
    pos2[:, 0, 1] = np.arange(0, 10, 1) + np.random.normal(0, 0.5, 10)
    # Add more NaNs
    pos2[3, 0, :] = np.nan
    pos2[7, 0, :] = np.nan

    # Create xarray datasets
    ds1 = xr.Dataset(
        data_vars={
            "position": (["time", "keypoints", "space"], pos1),
            "confidence": (["time", "keypoints"], np.ones((10, 1))),
        },
        coords={
            "time": time1,
            "keypoints": ["centroid"],
            "space": ["x", "y"],
            "individuals": ["individual_0"],
        },
    )

    ds2 = xr.Dataset(
        data_vars={
            "position": (["time", "keypoints", "space"], pos2),
            "confidence": (["time", "keypoints"], np.ones((10, 1))),
        },
        coords={
            "time": time2,
            "keypoints": ["centroid"],
            "space": ["x", "y"],
            "individuals": ["individual_0"],
        },
    )

    return [ds1, ds2]


def test_align_datasets(mock_datasets):
    """Test aligning datasets with different time points."""
    aligned = align_datasets(mock_datasets, interpolate=False)
    
    # Check that both arrays have the same time coordinates
    assert aligned[0].time.equals(aligned[1].time)
    
    # Check that NaNs are preserved when interpolate=False
    assert np.isnan(aligned[0].sel(time=3, space="x").values)
    assert np.isnan(aligned[1].sel(time=3, space="x").values)
    assert np.isnan(aligned[1].sel(time=7, space="x").values)
    
    # Test with interpolation
    aligned_interp = align_datasets(mock_datasets, interpolate=True)
    
    # Check that NaNs are interpolated
    assert not np.isnan(aligned_interp[0].sel(time=3, space="x").values)
    assert not np.isnan(aligned_interp[1].sel(time=3, space="x").values)
    assert not np.isnan(aligned_interp[1].sel(time=7, space="x").values)


def test_fuse_tracks_mean(mock_datasets):
    """Test mean fusion method."""
    aligned = align_datasets(mock_datasets, interpolate=True)
    fused = fuse_tracks_mean(aligned)
    
    # Check output dimensions
    assert "source" not in fused.dims
    assert "time" in fused.dims
    assert "space" in fused.dims
    
    # Check that the fused track has all time points
    assert len(fused.time) == 10
    
    # No NaNs when both sources are interpolated
    assert not np.isnan(fused).any()


def test_fuse_tracks_median(mock_datasets):
    """Test median fusion method."""
    aligned = align_datasets(mock_datasets, interpolate=True)
    fused = fuse_tracks_median(aligned)
    
    # Check output dimensions
    assert "source" not in fused.dims
    assert "time" in fused.dims
    assert "space" in fused.dims
    
    # No NaNs when both sources are interpolated
    assert not np.isnan(fused).any()


def test_fuse_tracks_weighted(mock_datasets):
    """Test weighted fusion method."""
    aligned = align_datasets(mock_datasets, interpolate=True)
    
    # Test with static weights
    weights = [0.7, 0.3]
    fused = fuse_tracks_weighted(aligned, weights=weights)
    
    # Check output dimensions
    assert "source" not in fused.dims
    assert "time" in fused.dims
    assert "space" in fused.dims
    
    # No NaNs when both sources are interpolated
    assert not np.isnan(fused).any()
    
    # Test with invalid weights (sum != 1)
    with pytest.raises(ValueError):
        fuse_tracks_weighted(aligned, weights=[0.5, 0.2])
    
    # Test with mismatched weights length
    with pytest.raises(ValueError):
        fuse_tracks_weighted(aligned, weights=[0.5, 0.3, 0.2])


def test_fuse_tracks_reliability(mock_datasets):
    """Test reliability-based fusion method."""
    aligned = align_datasets(mock_datasets, interpolate=False)  # Keep NaNs for testing
    
    # Test with automatic reliability metrics
    fused = fuse_tracks_reliability(aligned)
    
    # Check output dimensions
    assert "source" not in fused.dims
    assert "time" in fused.dims
    assert "space" in fused.dims
    
    # Test with custom reliability metrics
    reliability_metrics = [0.9, 0.5]  # First source more reliable
    fused = fuse_tracks_reliability(aligned, reliability_metrics=reliability_metrics)
    
    # Check that we still get a value for time point 7 where only source 1 has data
    assert not np.isnan(fused.sel(time=7, space="x").values)
    
    # Test with invalid window size (even number)
    with pytest.raises(ValueError):
        fuse_tracks_reliability(aligned, window_size=10)


def test_fuse_tracks_kalman(mock_datasets):
    """Test Kalman filter fusion method."""
    aligned = align_datasets(mock_datasets, interpolate=False)  # Keep NaNs for testing
    
    # Test with default parameters
    fused = fuse_tracks_kalman(aligned)
    
    # Check output dimensions
    assert "source" not in fused.dims
    assert "time" in fused.dims
    assert "space" in fused.dims
    
    # Kalman filter should interpolate over missing values
    assert not np.isnan(fused).any()
    
    # Test with custom parameters
    fused = fuse_tracks_kalman(
        aligned,
        process_noise_scale=0.1,
        measurement_noise_scales=[0.1, 0.5]
    )
    
    # Check that we get a smoother trajectory (less variance)
    x_vals = fused.sel(space="x").values
    diff = np.diff(x_vals)
    assert np.std(diff) < 0.5  # Standard deviation of the differences should be low
    
    # Test with mismatched noise scales length
    with pytest.raises(ValueError):
        fuse_tracks_kalman(aligned, measurement_noise_scales=[0.1, 0.2, 0.3])


def test_fuse_tracks_high_level(mock_datasets):
    """Test the high-level fuse_tracks interface."""
    # Test each method through the high-level interface
    methods = ["mean", "median", "weighted", "reliability", "kalman"]
    
    for method in methods:
        fused = fuse_tracks(
            datasets=mock_datasets,
            method=method,
            keypoint="centroid",
            interpolate_gaps=True
        )
        
        # Check output dimensions
        assert "time" in fused.dims
        assert "space" in fused.dims
        assert len(fused.space) == 2
        
        # No NaNs when interpolation is used
        assert not np.isnan(fused).any()
    
    # Test with invalid method
    with pytest.raises(ValueError):
        fuse_tracks(mock_datasets, method="invalid_method")
    
    # Test with non-existent keypoint
    with pytest.raises(ValueError):
        fuse_tracks(mock_datasets, keypoint="non_existent") 