import pytest
import xarray as xr


class TestMoveAccessor:
    """Test suite for the move_accessor module."""

    def test_property_with_valid_dataset(
        self, valid_pose_dataset, kinematic_property
    ):
        """Test that accessing a property of a valid pose dataset
        returns an instance of xr.DataArray with the correct name,
        and that the input xr.Dataset now contains the property as
        a data variable."""
        result = getattr(valid_pose_dataset.move, kinematic_property)
        assert isinstance(result, xr.DataArray)
        assert result.name == kinematic_property
        assert kinematic_property in valid_pose_dataset.data_vars

    def test_property_with_invalid_dataset(
        self, invalid_pose_dataset, kinematic_property
    ):
        """Test that accessing a property of an invalid pose dataset
        raises a ValueError."""
        expected_exception = (
            ValueError
            if isinstance(invalid_pose_dataset, xr.Dataset)
            else AttributeError
        )
        with pytest.raises(expected_exception):
            getattr(invalid_pose_dataset.move, kinematic_property)
