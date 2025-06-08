"""Test saving movement datasets to NetCDF files."""

import pandas as pd
import pytest
import xarray as xr

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


@pytest.mark.parametrize(
    "dataset",
    [
        "valid_poses_dataset",
        "valid_poses_dataset_with_nan",
        "valid_bboxes_dataset",  # time unit is in frames
        "valid_bboxes_dataset_in_seconds",
        "valid_bboxes_dataset_with_nan",
        "processed_dataset",
        "dataset_with_derived_variables",
        "dataset_with_datetime_index",
    ],
)
@pytest.mark.parametrize("engine", ["netcdf4", "scipy", "h5netcdf"])
def test_ds_save_and_load_netcdf(dataset, engine, tmp_path, request):
    """Test that saving a movement dataset to a NetCDF file and then
    loading it back returns the same Dataset.

    We test across all 3 NetCDF engines supported by xarray.
    """
    ds = request.getfixturevalue(dataset)
    netcdf_file = tmp_path / "test_dataset.nc"
    ds.to_netcdf(netcdf_file, engine=engine)
    loaded_ds = xr.load_dataset(netcdf_file)
    xr.testing.assert_allclose(loaded_ds, ds)
    assert loaded_ds.attrs == ds.attrs


def test_da_save_and_load_netcdf(valid_poses_dataset, tmp_path):
    """Test saving a DataArray to a NetCDF file and loading it back."""
    da = valid_poses_dataset["position"]
    netcdf_file = tmp_path / "test_dataarray.nc"
    da.to_netcdf(netcdf_file)
    loaded_da = xr.load_dataarray(netcdf_file)
    xr.testing.assert_allclose(loaded_da, da)
    assert loaded_da.attrs == da.attrs
