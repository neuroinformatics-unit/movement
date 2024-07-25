import pytest
import xarray as xr


class TestMovementDataset:
    """Test suite for the MovementDataset class."""

    @pytest.mark.parametrize(
        "valid_dataset", ("valid_poses_dataset", "valid_bboxes_dataset")
    )
    def test_compute_kinematics_with_valid_dataset(
        self, valid_dataset, kinematic_property, request
    ):
        """Test that computing a kinematic property of a valid
        pose dataset via accessor methods returns an instance of
        xr.DataArray.
        """
        valid_input_dataset = request.getfixturevalue(valid_dataset)

        result = getattr(
            valid_input_dataset.move, f"compute_{kinematic_property}"
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
    def test_invalid_method_call(self, valid_dataset, method, request):
        """Test that invalid accessor method calls raise an AttributeError."""
        valid_input_dataset = request.getfixturevalue(valid_dataset)
        with pytest.raises(AttributeError):
            getattr(valid_input_dataset.move, method)()
