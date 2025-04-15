import numpy as np
import pytest
import xarray as xr

from movement.utils.reports import report_nan_values

expectations = {
    "full_dataset-poses": {"expected": ["No missing points"]},
    "full_dataset-bboxes": {"expected": ["No missing points"]},
    "full_dataset-poses_with_nan": {
        "expected": [
            "position",
            "keypoints",
            "centroid",
            "left",
            "right",
            "individuals",
            "id_0",
            "3/10",
            "1/10",
            "10/10",
        ],
        "not_expected": ["id_1"],
    },
    "full_dataset-bboxes_with_nan": {
        "expected": ["position", "individuals", "id_0", "3/10"],
        "not_expected": ["id_1"],
    },
    "ind_dim_with_ndim_0-poses_with_nan": {
        "expected": ["centroid", "left", "right", "3/10", "1/10", "10/10"],
        "not_expected": ["id_0"],
    },
    "ind_dim_with_ndim_0-bboxes_with_nan": {
        "expected": ["3/10"],
        "not_expected": ["id_0"],
    },
    "kp_dim_with_ndim_0-poses_with_nan": {
        "expected": ["id_0", "3/10"],
        "not_expected": ["centroid"],
    },
    "both_dims_with_ndim_0-poses_with_nan": {
        "expected": ["3/10"],
        "not_expected": ["centroid", "id_0"],
    },
    "arbitrary_dims-poses_with_nan": {
        "expected": ["position", "other", "x", "y"]
    },
    "arbitrary_dims-simple_array_with_nan": {
        "expected": ["input", "1/3"],
    },
}  # If ndim=0, the coords are not explicitly reported


@pytest.mark.parametrize(
    "data, modifying_fn",
    [
        pytest.param(
            "valid_poses_dataset", lambda ds: ds, id="full_dataset-poses"
        ),
        pytest.param(
            "valid_bboxes_dataset", lambda ds: ds, id="full_dataset-bboxes"
        ),
        pytest.param(
            "valid_poses_dataset_with_nan",
            lambda ds: ds,
            id="full_dataset-poses_with_nan",
        ),
        pytest.param(
            "valid_bboxes_dataset_with_nan",
            lambda ds: ds,
            id="full_dataset-bboxes_with_nan",
        ),
        pytest.param(
            "valid_poses_dataset_with_nan",
            lambda ds: ds.isel(individuals=0),
            id="ind_dim_with_ndim_0-poses_with_nan",
        ),  # individuals dim is scalar
        pytest.param(
            "valid_bboxes_dataset_with_nan",
            lambda ds: ds.isel(individuals=0),
            id="ind_dim_with_ndim_0-bboxes_with_nan",
        ),  # individuals dim is scalar
        pytest.param(
            "valid_poses_dataset_with_nan",
            lambda ds: ds.isel(keypoints=0),
            id="kp_dim_with_ndim_0-poses_with_nan",
        ),  # keypoints dim is scalar (poses only)
        pytest.param(
            "valid_poses_dataset_with_nan",
            lambda ds: ds.isel(individuals=0, keypoints=0),
            id="both_dims_with_ndim_0-poses_with_nan",
        ),  # both dims are scalar (poses only)
        pytest.param(
            "valid_poses_dataset_with_nan",
            lambda ds: ds.rename({"space": "other"}),
            id="arbitrary_dims-poses_with_nan",
        ),  # count NaNs separately for x and y
        pytest.param(
            "simple_data_array_with_nan",
            lambda: xr.DataArray([1, np.nan, 3], dims="time"),
            id="arbitrary_dims-simple_array_with_nan",
        ),  # generic data array with required `time` dim
    ],
)
def test_report_nan_values(data, modifying_fn, request):
    """Test that the nan-value reporting function handles data with
    NaN values and that the report contains the correct NaN counts,
    keypoints, and individuals.
    """
    da = (
        modifying_fn()
        if data == "simple_data_array_with_nan"
        else modifying_fn(request.getfixturevalue(data).position)
    )
    report_str = report_nan_values(da)
    components = expectations.get(request.node.callspec.id)
    assert_components_in_report(components, report_str)


def assert_components_in_report(components, report_str):
    """Assert the expected components are in the report string."""
    assert all(
        component in report_str for component in components.get("expected", [])
    ) and all(
        component not in report_str
        for component in components.get("not_expected", [])
    )
