import re
from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
import xarray as xr

from movement.validators.datasets import (
    ValidBboxesInputs,
    ValidPosesInputs,
    _BaseDatasetInputs,
    _convert_fps_to_none_if_invalid,
    _convert_to_list_of_str,
)


@pytest.mark.parametrize(
    "input, expected_context, expected_output",
    [
        (
            "a",
            pytest.warns(
                UserWarning,
                match="Expected a list of strings",
            ),
            ["a"],
        ),
        (("a", "b", "c"), does_not_raise(), ["a", "b", "c"]),
        ([1, 2, 3], does_not_raise(), ["1", "2", "3"]),
        (
            5,
            pytest.raises(ValueError, match="Expected a list of strings"),
            None,
        ),
    ],
)
def test_convert_to_list_of_str(input, expected_context, expected_output):
    """Test conversion of various inputs to list of strings."""
    with expected_context:
        out = _convert_to_list_of_str(input)
        assert out == expected_output


@pytest.mark.parametrize(
    "input, expected_context, expected_output",
    [
        (
            -1,
            pytest.warns(UserWarning, match="Invalid .* Setting fps to None"),
            None,
        ),
        (
            0,
            pytest.warns(UserWarning, match="Invalid .* Setting fps to None"),
            None,
        ),
        (10.0, does_not_raise(), 10.0),
        (10, does_not_raise(), 10.0),
        (None, does_not_raise(), None),
    ],
)
def test_convert_fps_to_none_if_invalid(
    input, expected_context, expected_output
):
    """Test handling of valid and invalid fps values."""
    with expected_context:
        out = _convert_fps_to_none_if_invalid(input)
    assert out == expected_output


class TestBaseDatasetInputs:
    """Test the _BaseDatasetInputs class."""

    class StubAttr:
        """Stub for attrs.Attribute."""

        def __init__(self, name="stub_attribute"):
            """Initialise with a name."""
            self.name = name

    @staticmethod
    def make_stub_dataset_inputs_class(
        dim_names=("time", "space"), var_names=("position",)
    ):
        """Create a minimal subclass of _BaseDatasetInputs."""

        class StubDatasetInputs(_BaseDatasetInputs):
            """Minimal subclass for testing _BaseDatasetInputs."""

            DIM_NAMES = dim_names
            VAR_NAMES = var_names
            _ALLOWED_SPACE_DIM_SIZE = 2  # 2D positions

            def to_dataset(self) -> xr.Dataset:
                """Unimplemented stub method."""
                ...

        return StubDatasetInputs

    @pytest.fixture
    def valid_base_dataset(self):
        """Return a minimal valid base dataset."""
        return xr.Dataset(
            data_vars={
                "position": xr.DataArray(
                    np.zeros((5, 2)), dims=("time", "space")
                ),
            },
        )

    @pytest.fixture
    def missing_var_dataset(self, valid_base_dataset):
        """Return an invalid base dataset missing the required
        position variable.
        """
        return valid_base_dataset.drop_vars("position")

    @pytest.fixture
    def missing_dim_dataset(self, valid_base_dataset):
        """Return an invalid base dataset missing the required
        time dimension.
        """
        return valid_base_dataset.rename({"time": "tame"})

    def test_required_fields_missing(self):
        """Test that TypeError is raised when required fields are missing."""
        stub_dataset_inputs_class = self.make_stub_dataset_inputs_class()
        with pytest.raises(
            TypeError,
            match="missing 1 required keyword-only argument: 'position_array'",
        ):
            stub_dataset_inputs_class()

    @pytest.mark.parametrize(
        "dim_names, position_array, expected_confidence, expected_ind_names",
        [
            (("time", "space"), np.zeros((5, 2)), np.full(5, np.nan), None),
            (
                ("time", "space", "individual"),
                np.zeros((5, 2, 1)),
                np.full((5, 1), np.nan),
                ["id_0"],
            ),
            (
                ("time", "space", "individual"),
                np.zeros((5, 2, 3)),
                np.full((5, 3), np.nan),
                ["id_0", "id_1", "id_2"],
            ),
        ],
    )
    def test_optional_fields_defaults(
        self,
        dim_names,
        position_array,
        expected_confidence,
        expected_ind_names,
    ):
        """Test default values for optional fields."""
        stub_dataset_inputs_class = self.make_stub_dataset_inputs_class(
            dim_names=dim_names
        )
        data = stub_dataset_inputs_class(position_array=position_array)
        np.testing.assert_allclose(
            data.confidence_array, expected_confidence, equal_nan=True
        )
        assert data.individual_names == expected_ind_names
        assert data.fps is None
        assert data.source_software is None

    @pytest.mark.parametrize(
        "position_array, expected_error_message",
        [
            (np.zeros((5, 2, 3)), "have 2 dimensions, but got 3"),
            (np.zeros((5, 3)), "have 2 spatial dimensions, but got 3"),
        ],
        ids=[
            "Unexpected additional individual dimension",
            "Expect 2D positions but got 3D positions",
        ],
    )
    def test_position_array_with_mismatched_dimensions(
        self, position_array, expected_error_message
    ):
        """Test validation for position_array dimension mismatches."""
        stub_dataset_inputs_class = (
            self.make_stub_dataset_inputs_class()
        )  # time, 2D space
        with pytest.raises(
            ValueError,
            match=expected_error_message,
        ):
            stub_dataset_inputs_class(position_array=position_array)

    @pytest.mark.parametrize(
        "expected_shape, expected_exception",
        [
            ((2, 3), does_not_raise()),
            (
                (3, 2),
                pytest.raises(
                    ValueError,
                    match=re.escape("have shape (3, 2), but got (2, 3)"),
                ),
            ),
        ],
    )
    def test_validate_array_shape(self, expected_shape, expected_exception):
        """Test the _validate_array_shape static method directly."""
        stub_dataset_inputs_class = self.make_stub_dataset_inputs_class()
        val = np.zeros((2, 3))
        with expected_exception:
            stub_dataset_inputs_class._validate_array_shape(
                self.StubAttr(), val, expected_shape
            )

    @pytest.mark.parametrize(
        "val, expected_length, expected_exception",
        [
            (None, None, does_not_raise()),
            ([0, 1, 2], 3, does_not_raise()),
            (
                [0, 1],
                3,
                pytest.raises(
                    ValueError,
                    match="have length 3, but got 2",
                ),
            ),
        ],
    )
    def test_validate_list_length(
        self, val, expected_length, expected_exception
    ):
        """Test the _validate_list_length method directly."""
        stub_dataset_inputs_class = self.make_stub_dataset_inputs_class()
        with expected_exception:
            stub_dataset_inputs_class._validate_list_length(
                self.StubAttr(), val, expected_length
            )

    @pytest.mark.parametrize(
        "val, expected_exception",
        [
            (None, does_not_raise()),
            (["a", "b", "c"], does_not_raise()),
            (
                ["a", "b", "a"],
                pytest.raises(
                    ValueError,
                    match="are not unique",
                ),
            ),
        ],
    )
    def test_validate_list_uniqueness(self, val, expected_exception):
        """Test the _validate_list_uniqueness method directly."""
        stub_dataset_inputs_class = self.make_stub_dataset_inputs_class()
        with expected_exception:
            stub_dataset_inputs_class._validate_list_uniqueness(
                self.StubAttr(), val
            )

    @pytest.mark.parametrize(
        "dataset_fixture, expected_exception",
        [
            (
                "not_a_dataset",
                pytest.raises(TypeError, match="Expected an xarray Dataset"),
            ),
            (
                "missing_var_dataset",
                pytest.raises(
                    ValueError, match="Missing required data variables"
                ),
            ),
            (
                "missing_dim_dataset",
                pytest.raises(ValueError, match="Missing required dimensions"),
            ),
        ],
    )
    def test_validate(self, dataset_fixture, expected_exception, request):
        """Test the classmethod validate."""
        stub_dataset_inputs_class = self.make_stub_dataset_inputs_class()
        with expected_exception:
            stub_dataset_inputs_class.validate(
                request.getfixturevalue(dataset_fixture)
            )


class TestValidPosesInputs:
    """Test the ValidPosesInputs class."""

    @pytest.mark.parametrize(
        "position_array, keypoint_names, expected_context",
        [
            (np.zeros((5, 2, 1, 2)), None, does_not_raise(["keypoint_0"])),
            (
                np.zeros((5, 2, 3, 2)),
                None,
                does_not_raise(["keypoint_0", "keypoint_1", "keypoint_2"]),
            ),
            (
                np.zeros((5, 2, 3, 2)),
                ["a", "b", "c"],
                does_not_raise(["a", "b", "c"]),
            ),
            (
                np.zeros((5, 2, 3, 2)),
                ["a", "b"],
                pytest.raises(ValueError, match="length 3, but got 2"),
            ),
        ],
        ids=[
            "Single keypoint default name",
            "Multiple keypoints default names",
            "Explicit keypoint names",
            "Explicit keypoint names length mismatch",
        ],
    )
    def test_keypoint_names(
        self, position_array, keypoint_names, expected_context
    ):
        """Test keypoint_names validation in ValidPosesInputs."""
        with expected_context as expected_keypoint_names:
            data = ValidPosesInputs(
                position_array=position_array,
                keypoint_names=keypoint_names,
            )
            assert data.keypoint_names == expected_keypoint_names

    @pytest.mark.parametrize("fps", [30, None])
    def test_to_dataset(self, fps, valid_poses_dataset):
        """Test to_dataset creates the expected poses dataset."""
        ds = ValidPosesInputs(
            position_array=valid_poses_dataset.position.values,
            confidence_array=valid_poses_dataset.confidence.values,
            individual_names=valid_poses_dataset.individual.values,
            keypoint_names=valid_poses_dataset.keypoint.values,
            fps=fps,
            source_software=valid_poses_dataset.attrs["source_software"],
        ).to_dataset()
        if fps is None:
            xr.testing.assert_equal(ds, valid_poses_dataset)
        else:
            expected_ds = valid_poses_dataset.assign_coords(
                time=valid_poses_dataset.time.values / fps
            )
            expected_ds.time.attrs["units"] = "seconds"
            xr.testing.assert_equal(ds, expected_ds)


class TestValidBboxesInputs:
    """Test the ValidBboxesInputs class."""

    @pytest.mark.parametrize(
        "shape_array, expected_context",
        [
            (
                np.zeros((5, 2, 3)),
                does_not_raise(),
            ),
            (
                np.zeros((5, 2, 1)),
                pytest.raises(
                    ValueError,
                    match=re.escape("have shape (5, 2, 3), but got (5, 2, 1)"),
                ),
            ),
            (
                np.zeros((5, 2)),
                pytest.raises(
                    ValueError,
                    match="have 3 dimensions, but got 2",
                ),
            ),
            (
                np.zeros((5, 3, 1)),
                pytest.raises(
                    ValueError,
                    match="have 2 spatial dimensions, but got 3",
                ),
            ),
        ],
        ids=[
            "Valid shape_array",
            "Mismatch with position_array individual dimension",
            "Missing one dimension",
            "Expect 2D (width, height) shape but got 3D",
        ],
    )
    def test_shape_array(self, shape_array, expected_context):
        """Test shape_array validation."""
        position_array = np.zeros((5, 2, 3))  # time, space, individuals
        with expected_context:
            ValidBboxesInputs(
                position_array=position_array,
                shape_array=shape_array,
            )

    @pytest.mark.parametrize(
        "frame_array, expected_context",
        [
            (None, does_not_raise(np.arange(5)[:, None])),
            (
                np.arange(3, 8)[:, None],
                does_not_raise(np.arange(3, 8)[:, None]),
            ),
            (
                np.arange(3, 12, 2)[:, None],
                does_not_raise(np.arange(3, 12, 2)[:, None]),
            ),
            (
                np.arange(5)[::-1][:, None],
                pytest.raises(
                    ValueError,
                    match="not monotonically increasing",
                ),
            ),
            (
                np.zeros((5, 2)),
                pytest.raises(
                    ValueError,
                    match=re.escape("have shape (5, 1), but got (5, 2)"),
                ),
            ),
            (
                np.zeros((7, 1)),
                pytest.raises(
                    ValueError,
                    match=re.escape("have shape (5, 1), but got (7, 1)"),
                ),
            ),
        ],
        ids=[
            "Not provided, use default",
            "Consecutive monotonically increasing",
            "Non-consecutive but monotonically increasing",
            "Non-monotonically increasing frame numbers",
            "Shape mismatch: not a column vector",
            "Shape mismatch: frame number different from position_array",
        ],
    )
    def test_frame_array(self, frame_array, expected_context):
        """Test frame_array validation."""
        position_array = np.zeros((5, 2, 3))  # time, space, individuals
        shape_array = np.zeros((5, 2, 3))
        with expected_context as expected_frame_array:
            data = ValidBboxesInputs(
                position_array=position_array,
                shape_array=shape_array,
                frame_array=frame_array,
            )
            np.testing.assert_array_equal(
                data.frame_array, expected_frame_array
            )

    @pytest.mark.parametrize("fps", [30, None])
    def test_to_dataset(self, fps, valid_bboxes_dataset):
        """Test to_dataset creates the expected bboxes dataset."""
        ds = ValidBboxesInputs(
            position_array=valid_bboxes_dataset.position.values,
            shape_array=valid_bboxes_dataset.shape.values,
            confidence_array=valid_bboxes_dataset.confidence.values,
            individual_names=valid_bboxes_dataset.individual.values,
            fps=fps,
            source_software=valid_bboxes_dataset.attrs["source_software"],
        ).to_dataset()
        if fps is None:
            xr.testing.assert_equal(ds, valid_bboxes_dataset)
        else:
            expected_ds = valid_bboxes_dataset.assign_coords(
                time=valid_bboxes_dataset.time.values / fps
            )
            expected_ds.time.attrs["units"] = "seconds"
            xr.testing.assert_equal(ds, expected_ds)
