from collections import deque

import pytest

from movement.utils.reports import report_nan_values


@pytest.mark.parametrize(
    "valid_dataset",
    [
        "valid_poses_dataset",
        "valid_bboxes_dataset",
        "valid_poses_dataset_with_nan",
        "valid_bboxes_dataset_with_nan",
    ],
)
def test_report_nan_values_valid_dataset(
    valid_dataset,
    request,
):
    """Test that the nan-value reporting function handles valid
    poses and bboxes datasets.
    """
    ds = request.getfixturevalue(valid_dataset)
    da = ds.position
    report_str = report_nan_values(da)
    assert_components_in_report(
        {"expected": ["position", "individuals", "id_0", "id_1"]}, report_str
    )


@pytest.mark.parametrize(
    "valid_dataset",
    [
        "valid_poses_dataset",
        "valid_bboxes_dataset",
        "valid_poses_dataset_with_nan",
        "valid_bboxes_dataset_with_nan",
    ],
)
def test_report_nan_values_arbitrary_dims(valid_dataset, request):
    """Test that the nan-value reporting function handles data with
    arbitrary dimensions by checking the report contains NaN counts for
    x and y separately.
    """
    ds = request.getfixturevalue(valid_dataset)
    da = ds.rename({"space": "other"}).position
    report_str = report_nan_values(da)
    assert_components_in_report(
        {"expected": ["position", "other", "x", "y"]}, report_str
    )


expectations = {
    "ind_dim_with_ndim_0-valid_poses_dataset_with_nan": {
        "expected": ["centroid", "3/10", "left", "1/10", "right", "10/10"],
        "not_expected": ["id_0", "id_1"],
    },  # if only 1 ind exists, its name is not explicitly reported
    "ind_dim_with_ndim_0-valid_bboxes_dataset_with_nan": {
        "expected": ["3/10"],
        "not_expected": ["id_0", "id_1", "centroid"],
    },
    "kp_dim_with_ndim_0-valid_poses_dataset_with_nan": {
        "expected": ["id_0", "3/10", "id_1", "0/10"],
        "not_expected": ["centroid"],
    },
    "both_dims_with_ndim_0-valid_poses_dataset_with_nan": {
        "expected": ["3/10"],
        "not_expected": ["centroid", "id_0"],
    },
}


@pytest.mark.parametrize(
    "valid_dataset",
    ["valid_poses_dataset_with_nan", "valid_bboxes_dataset_with_nan"],
)
@pytest.mark.parametrize(
    "selection_fn",
    [
        # individuals dim is scalar
        lambda position: position.isel(individuals=0),
        # keypoints dim is scalar (poses only)
        lambda position: position.isel(keypoints=0),
        # both individuals and keypoints dims are scalar (poses only)
        lambda position: position.isel(individuals=0, keypoints=0),
    ],
    ids=[
        "ind_dim_with_ndim_0",
        "kp_dim_with_ndim_0",
        "both_dims_with_ndim_0",
    ],
)
def test_report_nan_values_scalar_dims(valid_dataset, selection_fn, request):
    """Test that the nan-value reporting function handles data with
    scalar ``individuals`` and/or ``keypoints`` dimensions (i.e.
    ``ndim=0``; e.g. when data is selected using ``isel`` or ``sel``).
    """
    components = expectations.get(request.node.callspec.id)
    if components:  # Skip tests with keypoints dim for bboxes
        ds = request.getfixturevalue(valid_dataset)
        da = selection_fn(ds.position)
        report_str = report_nan_values(da)
        assert_components_in_report(components, report_str)


def assert_components_in_report(components, report_str):
    """Assert the expected components are in the report string."""
    # tokenize report string
    report_str = deque(report_str.split())
    assert all(
        component in report_str for component in components.get("expected", [])
    ) and all(
        component not in report_str
        for component in components.get("not_expected", [])
    )
