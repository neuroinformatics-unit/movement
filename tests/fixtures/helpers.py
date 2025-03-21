"""Helpers fixture for ``movement`` test modules."""

import pytest
import xarray as xr


class Helpers:
    """General helper methods for ``movement`` test modules."""

    @staticmethod
    def assert_valid_dataset(dataset, expected_values):
        """Assert the dataset is a valid ``movement`` Dataset.

        The validation includes:
        - checking the dataset is an xarray Dataset
        - checking the expected variables are present and are of the right
          shape and type
        - checking the confidence array shape matches the position array
        - checking the dimensions and coordinates against the expected values
        - checking the coordinates' names and size
        - checking the metadata attributes

        Parameters
        ----------
        dataset : xr.Dataset
            The dataset to validate.
        expected_values : dict
            A dictionary containing the expected values for the dataset.
            It must contain the following keys:

            - dim_names: list of expected dimension names as defined in
              movement.validators.datasets
            - vars_dims: dictionary of data variable names and the
              corresponding dimension sizes

            Optional keys include:

            - file_path: Path to the source file
            - fps: int, frames per second
            - source_software: str, name of the software used to generate
              the dataset

        """
        # Check dataset is an xarray Dataset
        assert isinstance(dataset, xr.Dataset)

        # Expected variables are present and of right shape/type
        for var, ndim in expected_values.get("vars_dims").items():
            data_var = dataset.get(var)
            assert isinstance(data_var, xr.DataArray)
            assert data_var.ndim == ndim
        position_shape = dataset.position.shape

        # Confidence has the same shape as position, except for the space dim
        assert (
            dataset.confidence.shape == position_shape[:1] + position_shape[2:]
        )

        # Check the dims and coords
        expected_dim_names = expected_values.get("dim_names")
        expected_dim_length_dict = dict(
            zip(expected_dim_names, position_shape, strict=True)
        )
        assert expected_dim_length_dict == dataset.sizes

        # Check the coords
        for dim in expected_dim_names[1:]:
            assert all(isinstance(s, str) for s in dataset.coords[dim].values)
        assert all(coord in dataset.coords["space"] for coord in ["x", "y"])

        # Check the metadata attributes
        expected_file_path = expected_values.get("file_path")
        source_file = getattr(dataset, "source_file", None)
        assert source_file == (
            expected_file_path.as_posix()
            if expected_file_path is not None
            else None
        )
        assert dataset.source_software == expected_values.get(
            "source_software"
        )
        fps = getattr(dataset, "fps", None)
        assert fps == expected_values.get("fps")

    @staticmethod
    def count_nans(da):
        """Count number of NaNs in a DataArray."""
        return da.isnull().sum().item()

    @staticmethod
    def count_consecutive_nans(da):
        """Count occurrences of consecutive NaNs in a DataArray."""
        return (da.isnull().astype(int).diff("time") != 0).sum().item()


@pytest.fixture
def helpers():
    """Return an instance of the ``Helpers`` class."""
    return Helpers
