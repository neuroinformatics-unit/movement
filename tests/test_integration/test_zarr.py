"""Test saving movement datasets to Zarr stores."""

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
def test_ds_save_and_load_zarr(dataset, tmp_path, request):
    """Test that saving a movement dataset to a Zarr store and then
    loading it back returns the same Dataset.
    """
    ds = request.getfixturevalue(dataset)
    zarr_store = tmp_path / "test_dataset.zarr"
    ds.to_zarr(zarr_store, consolidated=True)
    loaded_ds = xr.open_zarr(zarr_store, consolidated=True)
    loaded_ds.load()
    xr.testing.assert_allclose(loaded_ds, ds)
    assert loaded_ds.attrs == ds.attrs


def test_da_save_and_load_zarr(valid_poses_dataset, tmp_path):
    """Test saving a DataArray to a Zarr store and loading it back."""
    da = valid_poses_dataset["position"]
    zarr_store = tmp_path / "test_dataarray.zarr"
    da.to_zarr(zarr_store, consolidated=True)
    loaded_da = xr.open_zarr(zarr_store, consolidated=True)["position"]
    loaded_da.load()
    xr.testing.assert_allclose(loaded_da, da)
    assert loaded_da.attrs == da.attrs
