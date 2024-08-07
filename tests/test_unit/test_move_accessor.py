from contextlib import nullcontext as does_not_raise

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


@pytest.mark.parametrize(
    "input_dataset, expected_exception, expected_patterns",
    (
        (
            "valid_poses_dataset",
            does_not_raise(),
            [],
        ),
        (
            "valid_bboxes_dataset",
            does_not_raise(),
            [],
        ),
        (
            "valid_bboxes_dataset_in_seconds",
            does_not_raise(),
            [],
        ),
        (
            "missing_dim_poses_dataset",
            pytest.raises(ValueError),
            ["Missing required dimensions:", "['time']"],
        ),
        (
            "missing_dim_bboxes_dataset",
            pytest.raises(ValueError),
            ["Missing required dimensions:", "['time']"],
        ),
        (
            "missing_two_dims_bboxes_dataset",
            pytest.raises(ValueError),
            ["Missing required dimensions:", "['space', 'time']"],
        ),
        (
            "missing_var_poses_dataset",
            pytest.raises(ValueError),
            ["Missing required data variables:", "['position']"],
        ),
        (
            "missing_var_bboxes_dataset",
            pytest.raises(ValueError),
            ["Missing required data variables:", "['position']"],
        ),
        (
            "missing_two_vars_bboxes_dataset",
            pytest.raises(ValueError),
            ["Missing required data variables:", "['position', 'shape']"],
        ),
    ),
)
def test_move_validate(
    input_dataset, expected_exception, expected_patterns, request
):
    """Test the validate method returns the expected message."""
    input_dataset = request.getfixturevalue(input_dataset)

    with expected_exception as excinfo:
        input_dataset.move.validate()

    if expected_patterns:
        error_message = str(excinfo.value)
        assert input_dataset.ds_type in error_message
        assert all([pattern in error_message for pattern in expected_patterns])
