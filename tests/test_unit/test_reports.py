import pytest

from movement.utils.reports import report_nan_values


@pytest.mark.parametrize(
    "data_selection",
    [
        lambda ds: ds.position,  # Entire dataset
        lambda ds: ds.position.sel(
            individuals="ind1"
        ),  # Missing "individuals" dim
        lambda ds: ds.position.sel(
            keypoints="key1"
        ),  # Missing "keypoints" dim
        lambda ds: ds.position.sel(
            individuals="ind1", keypoints="key1"
        ),  # Missing "individuals" and "keypoints" dims
    ],
)
def test_report_nan_values(
    capsys, valid_poses_dataset_with_nan, data_selection
):
    """Test that the nan-value reporting function handles data
    with missing ``individuals`` and/or ``keypoint`` dims, and
    that the dataset name is included in the report.
    """
    data = data_selection(valid_poses_dataset_with_nan)
    report_nan_values(data)
    out, _ = capsys.readouterr()
    assert data.name in out, "Dataset name should be in the output"
