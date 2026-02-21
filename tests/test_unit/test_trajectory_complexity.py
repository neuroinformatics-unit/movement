import numpy as np
import pytest
import xarray as xr

from movement.trajectory_complexity import compute_straightness_index

def test_straightness_index_straight_line():
    """Test that a perfectly straight diagonal line returns exactly 1.0"""
    data = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0]
    ])
    
    da = xr.DataArray(
        data,
        dims=["time", "space"],
        coords={"time": [0, 1, 2], "space": ["x", "y"]}
    )
    
    result = compute_straightness_index(da)
    
    assert np.isclose(result.item(), 1.0)

def test_straightness_index_loop():
    """Test that a path returning to its start returns 0.0"""
    data = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
        [0.0, 0.0]
    ])
    
    da = xr.DataArray(
        data,
        dims=["time", "space"],
        coords={"time": [0, 1, 2, 3, 4], "space": ["x", "y"]}
    )
    
    result = compute_straightness_index(da)
    
    assert np.isclose(result.item(), 0.0)