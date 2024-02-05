import xarray as xr


class TestMoveAccessor:
    """Test suite for the move_accessor module."""

    def test_property(self, valid_pose_dataset, kinematic_property):
        """Test that accessing a property returns an instance of
        xr.DataArray with the correct name, and that the input xr.Dataset
        now contains the property as a data variable."""
        result = getattr(valid_pose_dataset.move, kinematic_property)
        assert isinstance(result, xr.DataArray)
        assert result.name == kinematic_property
        assert kinematic_property in valid_pose_dataset.data_vars
