"""Test saving movement datasets to NetCDF files."""

import pytest
import xarray as xr

from movement.kinematics import compute_forward_vector, compute_speed


@pytest.fixture
def dataset_with_derived_variables(valid_poses_dataset):
    """Create a dataset with some derived variables."""
    ds = valid_poses_dataset.copy()
    ds["speed"] = compute_speed(ds["position"])
    ds["forward_vector"] = compute_forward_vector(
        ds["position"], "left", "right"
    )
    return ds


@pytest.mark.parametrize(
    "dataset",
    [
        "valid_poses_dataset",
        "valid_poses_dataset_with_nan",
        "valid_bboxes_dataset",  # time unit is in frames
        "valid_bboxes_dataset_in_seconds",
        "valid_bboxes_dataset_with_nan",
        "dataset_with_derived_variables",
    ],
)
@pytest.mark.parametrize("engine", ["netcdf4", "scipy", "h5netcdf"])
def test_save_and_load_netcdf(dataset, engine, tmp_path, request):
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
