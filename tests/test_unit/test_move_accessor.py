import pytest
import xarray as xr


class TestMoveAccessor:
    """Test suite for the move_accessor module."""

    @pytest.mark.parametrize("suffix", ["", "_pol"])
    def test_property_with_valid_dataset(
        self, valid_poses_dataset, kinematic_property, suffix
    ):
        """Test that accessing a kinematic property (in Cartesian
        or polar coordinates) of a valid pose dataset returns an
        instance of xr.DataArray with the correct name, and that
        the input xr.Dataset now contains the property as a data
        variable.
        """
        kinematic_property += suffix
        result = getattr(valid_poses_dataset.move, kinematic_property)
        assert isinstance(result, xr.DataArray)
        assert result.name == kinematic_property
        assert kinematic_property in valid_poses_dataset.data_vars

    def test_property_with_invalid_dataset(
        self, invalid_poses_dataset, kinematic_property
    ):
        """Test that accessing a property of an invalid pose dataset
        raises the appropriate error.
        """
        expected_exception = (
            ValueError
            if isinstance(invalid_poses_dataset, xr.Dataset)
            else AttributeError
        )
        with pytest.raises(expected_exception):
            getattr(invalid_poses_dataset.move, kinematic_property)
