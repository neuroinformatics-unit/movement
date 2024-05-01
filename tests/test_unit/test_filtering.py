import numpy as np
import pytest
import xarray as xr

from movement.filtering import (
    filter_by_confidence,
    interpolate_over_time,
    log_to_attrs,
)
from movement.sample_data import fetch_sample_data


@pytest.fixture(scope="module")
def sample_dataset():
    """Return a single-individual sample dataset."""
    return fetch_sample_data("DLC_single-mouse_EPM.predictions.h5")


def test_log_to_attrs(sample_dataset):
    """Test for the ``log_to_attrs()`` decorator. Decorates a mock function and
    checks that ``attrs`` contains all expected values.
    """

    @log_to_attrs
    def fake_func(ds, arg, kwarg=None):
        return ds

    ds = fake_func(sample_dataset, "test1", kwarg="test2")

    assert "log" in ds.attrs
    assert ds.attrs["log"][0]["operation"] == "fake_func"
    assert (
        ds.attrs["log"][0]["arg_1"] == "test1"
        and ds.attrs["log"][0]["kwarg"] == "test2"
    )


def test_interpolate_over_time(sample_dataset):
    """Test the ``interpolate_over_time`` function.

    Check that the number of nans is decreased after running this function
    on a filtered dataset
    """
    ds_filtered = filter_by_confidence(sample_dataset)
    ds_interpolated = interpolate_over_time(ds_filtered)

    def count_nans(ds):
        n_nans = np.count_nonzero(
            np.isnan(
                ds.position.sel(
                    individuals="individual_0", keypoints="snout"
                ).values[:, 0]
            )
        )
        return n_nans

    assert count_nans(ds_interpolated) < count_nans(ds_filtered)


def test_filter_by_confidence(sample_dataset, caplog):
    """Tests for the ``filter_by_confidence`` function.
    Checks that the function filters the expected amount of values
    from a known dataset, and tests that this value is logged
    correctly.
    """
    ds_filtered = filter_by_confidence(sample_dataset)

    assert isinstance(ds_filtered, xr.Dataset)

    n_nans = np.count_nonzero(
        np.isnan(
            ds_filtered.position.sel(
                individuals="individual_0", keypoints="snout"
            ).values[:, 0]
        )
    )
    assert n_nans == 2555

    # Check that diagnostics are being logged correctly
    assert f"snout: {n_nans}/{ds_filtered.time.values.shape[0]}" in caplog.text
