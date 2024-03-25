from contextlib import nullcontext as does_not_raise

import pytest
import xarray as xr

from movement.analysis import vector_utils


class TestAnalysis:
    """Test suite for the analysis module."""

    @pytest.mark.parametrize(
        "ds, expected_exception",
        [
            ("valid_poses_dataset", does_not_raise()),
            ("valid_poses_dataset_with_nan", does_not_raise()),
            ("missing_dim_dataset", pytest.raises(ValueError)),
        ],
    )
    def test_cart_and_pol_transform(
        self, ds, expected_exception, kinematic_property, request
    ):
        """Test transformation between Cartesian and polar coordinates
        with various kinematic properties."""
        ds = request.getfixturevalue(ds)
        with expected_exception:
            data = getattr(ds.move, kinematic_property)
            polar_data = vector_utils.cart2pol(data)
            cartesian_data = vector_utils.pol2cart(polar_data)
            xr.testing.assert_allclose(cartesian_data, data)
