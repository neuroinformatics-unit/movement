# from contextlib import nullcontext as does_not_raise

# import pytest
# import xarray as xr

# from movement.utils import vector


# class TestKinematicsVectorTransform:
#     """Test the vector transformation functionality with
#     various kinematic properties.
#     """

#     @pytest.mark.parametrize(
#         "ds, expected_exception",
#         [
#             ("valid_poses_dataset", does_not_raise()),
#             ("valid_poses_dataset_with_nan", does_not_raise()),
#             ("missing_dim_poses_dataset", pytest.raises(RuntimeError)),
#         ],
#     )
#     def test_cart_and_pol_transform(
#         self, ds, expected_exception, kinematic_property, request
#     ):
#         """Test transformation between Cartesian and polar coordinates
#         with various kinematic properties.
#         """
#         ds = request.getfixturevalue(ds)
#         with expected_exception:
#             data = getattr(ds.move, f"compute_{kinematic_property}")()
#             pol_data = vector.cart2pol(data)
#             cart_data = vector.pol2cart(pol_data)
#             xr.testing.assert_allclose(cart_data, data)
