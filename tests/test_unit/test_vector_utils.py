import numpy as np
import pytest
import xarray as xr

from movement.analysis import vector_utils


class TestVectorUtils:
    """Test suite for the vector_utils module."""

    @pytest.fixture
    def expected_dataset(self, valid_pose_dataset):
        """Return a function to generate the expected dataset
        for different kinematic properties."""

        def _expected_dataset(name):
            """Return an xarray.Dataset with expected norm and
            theta values."""
            dims = valid_pose_dataset.pose_tracks.dims[:-1]
            ds = xr.Dataset(
                data_vars={
                    "rho": xr.DataArray(
                        np.sqrt(
                            np.square(
                                valid_pose_dataset.pose_tracks.sel(space="x")
                            )
                            + np.square(
                                valid_pose_dataset.pose_tracks.sel(space="y")
                            )
                        ),
                        dims=dims,
                    ),
                    "theta": xr.DataArray(
                        np.full((10, 2, 2), 0.92729522),
                        dims=dims,
                    ),
                },
                coords={
                    "time": valid_pose_dataset.time,
                    "individuals": valid_pose_dataset.individuals,
                    "keypoints": valid_pose_dataset.keypoints,
                },
            )
            if name == "displacement":
                # Set the first values to zero
                ds.rho[0, :, :] = 0
                ds.theta[0, :, :] = 0
            elif name == "acceleration":
                # Set all values to zero
                ds.rho[:] = 0
                ds.theta[:] = 0
            return ds

        return _expected_dataset

    def test_cart2polar_(self):
        """Test rho computation."""
        x_vals = [5, 3.5355, 0, -10]
        y_vals = [0, 3.5355, 10, 0]
        time_coords = np.arange(len(x_vals), dtype=int)
        input_data = xr.DataArray(
            np.column_stack((x_vals, y_vals)),
            dims=["time", "space"],
            coords={"time": time_coords, "space": ["x", "y"]},
        )
        expected_theta = [0, 0.7854, 1.5708, 3.1416]
        expected_rho = [5.0000, 5.0000, 10.0000, 10.0000]
        expected = xr.DataArray(
            np.column_stack((expected_rho, expected_theta)),
            dims=["time", "space_polar"],
            coords={"time": time_coords, "space_polar": ["rho", "theta"]},
        )
        result = vector_utils.cart2polar(input_data)
        xr.testing.assert_allclose(result, expected)

    def test_cart2polar(
        self, valid_pose_dataset, kinematic_property, expected_dataset
    ):
        """Test rho computation for different kinematic properties."""
        data = getattr(valid_pose_dataset.move, kinematic_property)
        result = vector_utils.cart2polar(data)
        expected = xr.DataArray(
            np.sqrt(
                np.square(data.sel(space="x")) + np.square(data.sel(space="y"))
            ),
            dims=data.dims[:-1],
        )
        xr.testing.assert_allclose(
            result.sel(space_polar="rho", drop=True), expected
        )
