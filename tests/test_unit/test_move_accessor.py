import pytest
import xarray as xr


class TestMovementDataset:
    """Test suite for the MovementDataset class."""

    def test_compute_kinematics_with_valid_dataset(
        self, valid_poses_dataset, kinematic_property
    ):
        """Test that computing a kinematic property of a valid
        pose dataset via accessor methods returns an instance of
        xr.DataArray.
        """
        result = getattr(
            valid_poses_dataset.move, f"compute_{kinematic_property}"
        )()
        assert isinstance(result, xr.DataArray)

    def test_compute_kinematics_with_invalid_dataset(
        self, invalid_poses_dataset, kinematic_property
    ):
        """Test that computing a kinematic property of an invalid
        pose dataset via accessor methods raises the appropriate error.
        """
        with pytest.raises(AttributeError):
            getattr(
                invalid_poses_dataset.move, f"compute_{kinematic_property}"
            )()

    @pytest.mark.parametrize(
        "method", ["compute_invalid_property", "do_something"]
    )
    def test_invalid_compute(self, valid_poses_dataset, method):
        """Test that invalid accessor method calls raise an AttributeError."""
        with pytest.raises(AttributeError):
            getattr(valid_poses_dataset.move, method)()


class TestMovementDataArray:
    """Test suite for the MovementDataArray class."""

    @pytest.mark.parametrize(
        "comparator", ["lt", "le", "eq", "ne", "ge", "gt"]
    )
    def test_filter_with_valid_dataarray(
        self, valid_poses_dataset, comparator
    ):
        """Test that filtering a valid pose data array returns an
        instance of xr.DataArray.
        """
        position = valid_poses_dataset.position
        confidence = valid_poses_dataset.confidence
        result = getattr(position.move, f"filter_{comparator}")(confidence)
        assert isinstance(result, xr.DataArray)

    @pytest.mark.parametrize(
        "method", ["filter_invalid_comparator", "do_something"]
    )
    def test_invalid_method(self, valid_poses_dataset, method):
        """Test that calling an invalid method raises an AttributeError."""
        with pytest.raises(AttributeError):
            position = valid_poses_dataset.position
            confidence = valid_poses_dataset.confidence
            getattr(position.move, method)(confidence)
