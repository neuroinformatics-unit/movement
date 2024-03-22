import xarray as xr

from movement.analysis import vector_utils


class TestVectorUtils:
    """Test suite for the vector_utils module."""

    def test_cart2pol_pol2cart(self, valid_pose_dataset, kinematic_property):
        """Test transformation between Cartesian and polar coordinates."""
        data = getattr(valid_pose_dataset.move, kinematic_property)
        polar_data = vector_utils.cart2pol(data)
        cartesian_data = vector_utils.pol2cart(polar_data)
        xr.testing.assert_allclose(cartesian_data, data)

    # TODO: test with missing "space" dimension
    # TODO: test with missing "x" and "y" variables
    # TODO: test with Nan values
