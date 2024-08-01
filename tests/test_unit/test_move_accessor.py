import pytest
import xarray as xr


class TestMovementDataset:
    """Test suite for the MovementDataset class."""

    def test_compute_kinematics_with_valid_dataset(
        self, valid_poses_dataset, kinematic_property
    ):
        """Test that computing a kinematic property of a valid
        pose dataset via accessor methods returns an instance of
        xr.DataArray.
        """
        result = getattr(
            valid_poses_dataset.move, f"compute_{kinematic_property}"
        )()
        assert isinstance(result, xr.DataArray)

    def test_compute_kinematics_with_invalid_dataset(
        self, invalid_poses_dataset, kinematic_property
    ):
        """Test that computing a kinematic property of an invalid
        poses dataset via accessor methods raises the appropriate error.
        """
        expected_exception = (
            RuntimeError
            if isinstance(invalid_poses_dataset, xr.Dataset)
            else AttributeError
        )
        with pytest.raises(expected_exception):
            getattr(
                invalid_poses_dataset.move, f"compute_{kinematic_property}"
            )()


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
    "invalid_dataset, log_message",
    (
        (
            "missing_var_poses_dataset",
            "Missing required data variables: ['position']",
        ),
        (
            "missing_var_bboxes_dataset",
            "Missing required dimensions: ['keypoints']",
        ),
        ("missing_dim_poses_dataset", "Missing required dimensions: ['time']"),
        (
            "missing_dim_bboxes_dataset",
            "Missing required dimensions: ['keypoints', 'time']",
        ),
    ),
)
def test_move_validate(invalid_dataset, log_message, request):
    """Test the validate method returns the expected message."""
    invalid_dataset = request.getfixturevalue(invalid_dataset)

    with pytest.raises(ValueError) as excinfo:
        invalid_dataset.move.validate()
    assert log_message in str(excinfo.value)
