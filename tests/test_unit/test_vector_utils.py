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

        def _expected_dataset(input_type=None):
            """Return an xarray.Dataset with input and expected output."""
            x_vals = [1, 1, 0, -1, -10, -1, 0, 1, 5, 3.5355, 0, -2]
            y_vals = [0, 1, 1, 1, 0, -1, -1, -1, 0, 3.5355, 10, -2]
            time_coords = np.arange(len(x_vals))
            expected_theta = np.pi * np.array(
                [
                    0,
                    0.25,
                    0.5,
                    0.75,
                    1,
                    -0.75,
                    -0.5,
                    -0.25,
                    0,
                    0.25,
                    0.5,
                    -0.75,
                ]
            )
            expected_rho = np.sqrt(
                np.array(x_vals) ** 2 + np.array(y_vals) ** 2
            )
            if input_type == "multi_dim_input":
                n_individuals = valid_pose_dataset.coords["individuals"].size
                n_keypoints = valid_pose_dataset.coords["keypoints"].size
                n_frames = int(len(x_vals) / (n_individuals * n_keypoints))
                input_data = xr.DataArray(
                    np.column_stack((x_vals, y_vals)).reshape(
                        n_frames, n_individuals, n_keypoints, 2
                    ),
                    coords={
                        "time": np.arange(n_frames),
                        "individuals": valid_pose_dataset.coords[
                            "individuals"
                        ],
                        "keypoints": valid_pose_dataset.coords["keypoints"],
                        "space": valid_pose_dataset.coords["space"],
                    },
                )
                expected = xr.DataArray(
                    np.column_stack((expected_rho, expected_theta)).reshape(
                        n_frames, n_individuals, n_keypoints, 2
                    ),
                    coords={
                        "time": np.arange(n_frames),
                        "individuals": valid_pose_dataset.coords[
                            "individuals"
                        ],
                        "keypoints": valid_pose_dataset.coords["keypoints"],
                        "space_polar": ["rho", "theta"],
                    },
                )
            else:
                input_data = xr.DataArray(
                    np.column_stack((x_vals, y_vals)),
                    dims=["time", "space"],
                    coords={"time": time_coords, "space": ["x", "y"]},
                )
                expected = xr.DataArray(
                    np.column_stack((expected_rho, expected_theta)),
                    dims=["time", "space_polar"],
                    coords={
                        "time": time_coords,
                        "space_polar": ["rho", "theta"],
                    },
                )
            ds = xr.Dataset(
                data_vars={
                    "input": input_data,
                    "expected": expected,
                },
            )
            return ds

        return _expected_dataset

    @pytest.mark.parametrize(
        "input_type", ["single_dim_input", "multi_dim_input"]
    )
    def test_cart2pol_(self, expected_dataset, input_type):
        """Test cart2pol with known values."""
        params = expected_dataset(input_type)
        result = vector_utils.cart2pol(params.input)
        xr.testing.assert_allclose(result, params.expected)

    @pytest.mark.parametrize(
        "input_type", ["single_dim_input", "multi_dim_input"]
    )
    def test_pol2cart(self, expected_dataset, input_type):
        """Test cart2pol with known values."""
        params = expected_dataset(input_type)
        result = vector_utils.pol2cart(params.expected)
        xr.testing.assert_allclose(result, params.input)

    def test_cart2pol_pol2cart(self, valid_pose_dataset, kinematic_property):
        """Test transformation between Cartesian and polar coordinates."""
        data = getattr(valid_pose_dataset.move, kinematic_property)
        polar_data = vector_utils.cart2pol(data)
        cartesian_data = vector_utils.pol2cart(polar_data)

        xr.testing.assert_allclose(cartesian_data, data)

    # TODO: test with missing "space" dimension
    # TODO: test with missing "x" and "y" variables
    # TODO: test with Nan values
