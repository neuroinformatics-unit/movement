"""Test saving movement datasets to NetCDF files."""

import pytest
import xarray as xr


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
