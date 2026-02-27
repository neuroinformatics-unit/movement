from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
import xarray as xr

from movement.utils.reports import report_nan_values


def assert_components_in_report(components, report_str):
    """Assert the expected components are in the report string."""
    assert all(
        component in report_str for component in components.get("expected", [])
    ) and all(
        component not in report_str
        for component in components.get("not_expected", [])
    )


@pytest.mark.parametrize(
    "data, expectations",
    [
        ("valid_poses_dataset", {"expected": ["No missing points"]}),
        ("valid_bboxes_dataset", {"expected": ["No missing points"]}),
        (
            "valid_poses_dataset_with_nan",
            {
                "expected": [
                    "position",
                    "keypoint",
                    "centroid",
                    "left",
                    "right",
                    "individual",
                    "id_0",
                    "3/10",
                    "1/10",
                    "10/10",
                ],
                "not_expected": ["id_1"],
            },
        ),
        (
            "valid_bboxes_dataset_with_nan",
            {
                "expected": ["position", "individual", "id_0", "3/10"],
                "not_expected": ["id_1"],
            },
        ),
    ],
)
def test_report_nan_values_full_dataset(data, expectations, request):
    """Test that the nan-value reporting function handles full and
    valid data with or without NaN values and that the report contains
    the correct NaN counts, keypoints, and individuals.
    """
    da = request.getfixturevalue(data).position
    report_str = report_nan_values(da)
    assert_components_in_report(expectations, report_str)


@pytest.mark.parametrize(
    "data, selection_fn, expectations",
    [
        (
            "valid_poses_dataset_with_nan",
            lambda ds: ds.isel(individual=0),
            {
                "expected": [
                    "centroid",
                    "left",
                    "right",
                    "3/10",
                    "1/10",
                    "10/10",
                ],
                "not_expected": ["id_0"],
            },
        ),
        (
            "valid_bboxes_dataset_with_nan",
            lambda ds: ds.isel(individual=0),
            {
                "expected": ["3/10"],
                "not_expected": ["id_0"],
            },
        ),
        (
            "valid_poses_dataset_with_nan",
            lambda ds: ds.isel(keypoint=0),
            {
                "expected": ["id_0", "3/10"],
                "not_expected": ["centroid"],
            },
        ),
        (
            "valid_poses_dataset_with_nan",
            lambda ds: ds.isel(individual=0, keypoint=0),
            {
                "expected": ["3/10"],
                "not_expected": ["centroid", "id_0"],
            },
        ),
    ],
    ids=[
        "ind_dim_with_ndim_0-poses",  # individuals dim is scalar
        "ind_dim_with_ndim_0-bboxes",  # individuals dim is scalar
        "kp_dim_with_ndim_0-poses",  # keypoints dim is scalar
        "both_dims_with_ndim_0-poses",  # both dims are scalar
    ],
)  # If ndim=0, the dim coords are not explicitly reported
def test_report_nan_values_scalar_dims(
    data, selection_fn, expectations, request
):
    """Test that the nan-value reporting function handles data with
    scalar dimensions (i.e. dimension.ndim == 0), for example, when
    using ``isel()`` or ``sel()`` to select a single individual or
    keypoint.
    """
    da = selection_fn(request.getfixturevalue(data)).position
    report_str = report_nan_values(da)
    assert_components_in_report(expectations, report_str)


@pytest.mark.parametrize(
    "data, fetch_data, expectations",
    [
        (
            "valid_poses_dataset_with_nan",
            lambda ds: ds.rename({"space": "other"}),
            does_not_raise({"expected": ["position", "other", "x", "y"]}),
        ),  # count NaNs separately for x and y
        (
            "simple_data_array_with_nan",
            lambda: xr.DataArray([1, np.nan, 3], dims="time"),
            does_not_raise({"expected": ["data", "1/3"]}),
        ),  # generic data array with required time dim
        (
            "invalid_data_array_with_nan",
            lambda: xr.DataArray([1, np.nan, 3], dims="dim1"),
            pytest.raises(ValueError, match=".*must contain.*time.*"),
        ),  # invalid data array without required time dim
    ],
    ids=["separate_x_y_dims", "simple_data_array", "missing_time_dim"],
)
def test_report_nan_values_arbitrary_dims(
    data, fetch_data, expectations, request
):
    """Test that the nan-value reporting function handles data with
    arbitrary dimensions as long as the required `time` dimension is
    present.
    """
    da = (
        fetch_data(request.getfixturevalue(data).position)
        if data == "valid_poses_dataset_with_nan"
        else fetch_data()
    )
    with expectations as e:
        report_str = report_nan_values(da)
        assert_components_in_report(e, report_str)
