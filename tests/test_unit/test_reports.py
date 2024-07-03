from movement.utils.reports import report_nan_values


def test_report_nan_values(capsys, valid_poses_dataset_with_nan):
    """Test that the nan-value report contains the name of the dataset."""
    data = valid_poses_dataset_with_nan.position
    report_nan_values(data)
    out, _ = capsys.readouterr()
    assert data.name in out
