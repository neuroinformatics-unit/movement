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
        pose dataset via accessor methods raises the appropriate error.
        """
        with pytest.raises(AttributeError):
            getattr(
                invalid_poses_dataset.move, f"compute_{kinematic_property}"
            )()

    @pytest.mark.parametrize(
        "method", ["compute_invalid_property", "do_something"]
    )
    def test_invalid_compute(self, valid_poses_dataset, method):
        """Test that invalid accessor method calls raise an AttributeError."""
        with pytest.raises(AttributeError):
            getattr(valid_poses_dataset.move, method)()
