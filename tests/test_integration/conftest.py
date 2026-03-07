"""Shared fixtures for integration tests."""

import pandas as pd
import pytest

from movement.filtering import filter_by_confidence, rolling_filter
from movement.kinematics import compute_forward_vector, compute_speed
from movement.transforms import scale


@pytest.fixture
def processed_dataset(valid_poses_dataset):
    """Process a valid poses dataset by applying filters and transforms."""
    ds = valid_poses_dataset.copy()
    ds["position_filtered"] = filter_by_confidence(
        ds["position"], ds["confidence"], threshold=0.5
    )
    ds["position_smoothed"] = rolling_filter(
        ds["position"], window=3, min_periods=2, statistic="median"
    )
    ds["position_scaled"] = scale(
        ds["position_smoothed"], factor=1 / 10, space_unit="cm"
    )
    return ds


@pytest.fixture
def dataset_with_derived_variables(valid_poses_dataset):
    """Create a dataset with some derived variables."""
    ds = valid_poses_dataset.copy()
    ds["speed"] = compute_speed(ds["position"])
    ds["forward_vector"] = compute_forward_vector(
        ds["position"], "left", "right"
    )
    return ds


@pytest.fixture
def dataset_with_datetime_index(valid_poses_dataset):
    """Create a dataset with a pd.DateTimeIndex as the time coordinate."""
    ds = valid_poses_dataset.copy()
    timestamps = pd.date_range(
        start=pd.Timestamp.now(),
        periods=ds.sizes["time"],
        freq=pd.Timedelta(seconds=1),
    )
    ds.assign_coords(time=timestamps)
    return ds
