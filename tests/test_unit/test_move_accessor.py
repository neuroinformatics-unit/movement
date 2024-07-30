import pytest
import xarray as xr


@pytest.mark.parametrize(
    "valid_dataset", ("valid_poses_dataset", "valid_bboxes_dataset")
)
def test_compute_kinematics_with_valid_dataset(
    valid_dataset, kinematic_property, request
):
    """Test that computing a kinematic property of a valid
    poses or bounding boxes dataset via accessor methods returns
    an instance of xr.DataArray.
    """
    valid_input_dataset = request.getfixturevalue(valid_dataset)

    result = getattr(
        valid_input_dataset.move, f"compute_{kinematic_property}"
    )()
    assert isinstance(result, xr.DataArray)


@pytest.mark.parametrize(
    "invalid_dataset",
    (
        "not_a_dataset",
        "empty_dataset",
        "missing_var_poses_dataset",
        "missing_var_bboxes_dataset",
        "missing_dim_poses_dataset",
        "missing_dim_bboxes_dataset",
    ),
)
def test_compute_kinematics_with_invalid_dataset(
    invalid_dataset, kinematic_property, request
):
    """Test that computing a kinematic property of an invalid
    poses or bounding boxes dataset via accessor methods raises
    the appropriate error.
    """
    invalid_dataset = request.getfixturevalue(invalid_dataset)
    expected_exception = (
        RuntimeError
        if isinstance(invalid_dataset, xr.Dataset)
        else AttributeError
    )
    with pytest.raises(expected_exception):
        getattr(invalid_dataset.move, f"compute_{kinematic_property}")()


@pytest.mark.parametrize(
    "method", ["compute_invalid_property", "do_something"]
)
@pytest.mark.parametrize(
    "valid_dataset", ("valid_poses_dataset", "valid_bboxes_dataset")
)
def test_invalid_move_method_call(valid_dataset, method, request):
    """Test that invalid accessor method calls raise an AttributeError."""
    valid_input_dataset = request.getfixturevalue(valid_dataset)
    with pytest.raises(AttributeError):
        getattr(valid_input_dataset.move, method)()
