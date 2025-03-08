"""Unit tests for nan reporting utilities."""

import numpy as np
import pytest
import xarray as xr
from movement.utils.reports import calculate_nan_stats, report_nan_values

# Fixtures ----------------------------------------------------------------

@pytest.fixture
def full_dim_dataarray():
    """DataArray with all dimensions."""
    da = xr.DataArray(
        np.random.randn(2, 3, 100, 3),  # (individuals, keypoints, time, space)
        dims=("individuals", "keypoints", "time", "space"),
        coords={
            "individuals": ["mouse1", "mouse2"],
            "keypoints": ["snout", "paw", "tail"],
            "time": range(100),
            "space": ["x", "y", "z"]
        },
        name="full_dims"
    )
    da[0, 0, 0:5, :] = np.nan  # 5 NaNs for mouse1/snout
    return da

@pytest.fixture
def no_space_dataarray():
    """DataArray without space dimension."""
    da = xr.DataArray(
        np.random.randn(2, 3, 100),  # (individuals, keypoints, time)
        dims=("individuals", "keypoints", "time"),
        coords={
            "individuals": ["mouse1", "mouse2"],
            "keypoints": ["snout", "paw", "tail"],
            "time": range(100)
        },
        name="no_space"
    )
    da[0, 0, 0:10] = np.nan  # 10 NaNs for mouse1/snout
    return da

@pytest.fixture
def minimal_dataarray():
    """DataArray with only time dimension."""
    da = xr.DataArray(
        np.random.randn(100),
        dims=("time",),
        coords={"time": range(100)},
        name="minimal"
    )
    da[0:15] = np.nan  # 15 NaNs
    return da

# Tests -------------------------------------------------------------------

class TestCalculateNanStats:
    """Test suite for calculate_nan_stats function."""
    
    def test_full_dims(self, full_dim_dataarray):
        """Test calculation with all dimensions present."""
        result = calculate_nan_stats(full_dim_dataarray, "snout", "mouse1")
        assert "snout: 5/100 (5.0%)" in result

    def test_no_space(self, no_space_dataarray):
        """Test calculation without space dimension."""
        result = calculate_nan_stats(no_space_dataarray, "snout", "mouse1")
        assert "snout: 10/100 (10.0%)" in result

    def test_minimal(self, minimal_dataarray):
        """Test calculation with minimal data (time only)."""
        result = calculate_nan_stats(minimal_dataarray)
        assert "data: 15/100 (15.0%)" in result

class TestReportNanValues:
    """Test suite for report_nan_values function."""
    
    def test_full_dims(self, full_dim_dataarray):
        """Test report generation with all dimensions."""
        report = report_nan_values(full_dim_dataarray)
        assert "Individual: mouse1" in report
        assert "snout: 5/100 (5.0%)" in report
        assert "(any spatial coordinate)" in report

    def test_no_space(self, no_space_dataarray):
        """Test report without space dimension."""
        report = report_nan_values(no_space_dataarray)
        assert "Individual: mouse1" in report
        assert "snout: 10/100 (10.0%)" in report
        assert "(any spatial coordinate)" not in report

    def test_minimal(self, minimal_dataarray):
        """Test report with minimal data (time only)."""
        report = report_nan_values(minimal_dataarray)
        assert "data: 15/100 (15.0%)" in report
        assert "Individual:" not in report

if __name__ == "__main__":
    pytest.main([__file__, "-v"])