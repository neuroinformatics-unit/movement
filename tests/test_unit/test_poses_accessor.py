import pytest
import xarray as xr


class TestMoveAccessor:
    """Test suite for the move_accessor module."""

    @pytest.mark.parametrize(
        "property_name", ["displacement", "velocity", "acceleration"]
    )
    def test_property(self, valid_pose_dataset, property_name):
        """Test that the property returns an instance of xr.DataArray
        with the correct name, and that the input xr.Dataset contains
        the newly-added data variable with the same name."""
        result = getattr(valid_pose_dataset.move, property_name)
        assert isinstance(result, xr.DataArray)
        assert result.name == property_name
        assert property_name in valid_pose_dataset.data_vars
