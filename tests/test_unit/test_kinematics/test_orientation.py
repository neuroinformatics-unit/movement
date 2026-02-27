import re
from typing import Literal

import numpy as np
import pytest
import xarray as xr

from movement import kinematics


@pytest.fixture
def valid_data_array_for_forward_vector():
    """Return a position data array for an individual with 3 keypoints
    (left ear, right ear and nose), tracked for 4 frames, in x-y space.
    """
    time = [0, 1, 2, 3]
    individuals = ["id_0"]
    keypoints = ["left_ear", "right_ear", "nose"]
    space = ["x", "y"]

    ds = xr.DataArray(
        [
            [[[1, 0], [-1, 0], [0, -1]]],  # time 0
            [[[0, 1], [0, -1], [1, 0]]],  # time 1
            [[[-1, 0], [1, 0], [0, 1]]],  # time 2
            [[[0, -1], [0, 1], [-1, 0]]],  # time 3
        ],
        dims=["time", "individual", "keypoint", "space"],
        coords={
            "time": time,
            "individual": individuals,
            "keypoint": keypoints,
            "space": space,
        },
    )
    return ds


@pytest.fixture
def invalid_input_type_for_forward_vector(valid_data_array_for_forward_vector):
    """Return a numpy array of position values by individual, per keypoint,
    over time.
    """
    return valid_data_array_for_forward_vector.values


@pytest.fixture
def invalid_dimensions_for_forward_vector(valid_data_array_for_forward_vector):
    """Return a position DataArray in which the ``keypoint`` dimension has
    been dropped.
    """
    return valid_data_array_for_forward_vector.sel(keypoint="nose", drop=True)


@pytest.fixture
def invalid_spatial_dimensions_for_forward_vector(
    valid_data_array_for_forward_vector,
):
    """Return a position DataArray containing three spatial dimensions."""
    dataarray_3d = valid_data_array_for_forward_vector.pad(
        space=(0, 1), constant_values=0
    )
    return dataarray_3d.assign_coords(space=["x", "y", "z"])


@pytest.fixture
def valid_data_array_for_forward_vector_with_nan(
    valid_data_array_for_forward_vector,
):
    """Return a position DataArray where position values are NaN for the
    ``left_ear`` keypoint at time ``1``.
    """
    nan_dataarray = valid_data_array_for_forward_vector.where(
        (valid_data_array_for_forward_vector.time != 1)
        | (valid_data_array_for_forward_vector.keypoint != "left_ear")
    )
    return nan_dataarray


def test_compute_forward_vector(valid_data_array_for_forward_vector):
    """Test that the correct output forward direction vectors
    are computed from a valid mock dataset.
    """
    forward_vector = kinematics.compute_forward_vector(
        valid_data_array_for_forward_vector,
        "left_ear",
        "right_ear",
        camera_view="bottom_up",
    )
    forward_vector_flipped = kinematics.compute_forward_vector(
        valid_data_array_for_forward_vector,
        "left_ear",
        "right_ear",
        camera_view="top_down",
    )
    head_vector = kinematics.compute_head_direction_vector(
        valid_data_array_for_forward_vector,
        "left_ear",
        "right_ear",
        camera_view="bottom_up",
    )
    assert forward_vector.name == "forward_vector"
    assert forward_vector_flipped.name == "forward_vector"
    assert head_vector.name == "head_direction_vector"

    known_vectors = np.array([[[0, -1]], [[1, 0]], [[0, 1]], [[-1, 0]]])

    for output_array in [forward_vector, forward_vector_flipped, head_vector]:
        assert isinstance(output_array, xr.DataArray)
        for preserved_coord in ["time", "space", "individual"]:
            assert np.all(
                output_array[preserved_coord]
                == valid_data_array_for_forward_vector[preserved_coord]
            )
        assert set(output_array["space"].values) == {"x", "y"}
    assert np.equal(forward_vector.values, known_vectors).all()
    assert np.equal(forward_vector_flipped.values, known_vectors * -1).all()
    assert head_vector.equals(forward_vector)


@pytest.mark.parametrize(
    "input_data, expected_error, expected_match_str, keypoints",
    [
        (
            "invalid_input_type_for_forward_vector",
            TypeError,
            "must be an xarray.DataArray",
            ["left_ear", "right_ear"],
        ),
        (
            "invalid_dimensions_for_forward_vector",
            ValueError,
            "Input data must contain ['keypoint']",
            ["left_ear", "right_ear"],
        ),
        (
            "invalid_spatial_dimensions_for_forward_vector",
            ValueError,
            "must have exactly 2 spatial dimensions",
            ["left_ear", "right_ear"],
        ),
        (
            "valid_data_array_for_forward_vector",
            ValueError,
            "keypoints may not be identical",
            ["left_ear", "left_ear"],
        ),
    ],
)
def test_compute_forward_vector_with_invalid_input(
    input_data, keypoints, expected_error, expected_match_str, request
):
    """Test that ``compute_forward_vector`` catches errors
    correctly when passed invalid inputs.
    """
    # Get fixture
    input_data = request.getfixturevalue(input_data)

    # Catch error
    with pytest.raises(expected_error, match=re.escape(expected_match_str)):
        kinematics.compute_forward_vector(
            input_data, keypoints[0], keypoints[1]
        )


def test_nan_behavior_forward_vector(
    valid_data_array_for_forward_vector_with_nan,
):
    """Test that ``compute_forward_vector()`` generates the
    expected output for a valid input DataArray containing ``NaN``
    position values at a single time (``1``) and keypoint
    (``left_ear``).
    """
    nan_time = 1
    forward_vector = kinematics.compute_forward_vector(
        valid_data_array_for_forward_vector_with_nan, "left_ear", "right_ear"
    )
    # trunk-ignore(bandit/B101)
    assert forward_vector.name == "forward_vector"
    # Check coord preservation
    for preserved_coord in ["time", "space", "individual"]:
        assert np.all(
            forward_vector[preserved_coord]
            == valid_data_array_for_forward_vector_with_nan[preserved_coord]
        )
    assert set(forward_vector["space"].values) == {"x", "y"}
    # Should have NaN values in the forward vector at time 1 and left_ear
    nan_values = forward_vector.sel(time=nan_time)
    assert nan_values.shape == (1, 2)
    assert np.isnan(nan_values).all(), (
        "NaN values not returned where expected!"
    )
    # Should have no NaN values in the forward vector in other positions
    assert not np.isnan(
        forward_vector.sel(
            time=[
                t
                for t in valid_data_array_for_forward_vector_with_nan.time
                if t != nan_time
            ]
        )
    ).any()


class TestForwardVectorAngle:
    """Test the compute_forward_vector_angle function.

    These tests are grouped together into a class to distinguish them from the
    other methods that are tested in the Kinematics module.

    Note that since this method is a combination of calls to two lower-level
    methods, we run limited input/output checks in this collection.
    Correctness of the results is delegated to the tests of the dependent
    methods, as appropriate.
    """

    x_axis = np.array([1.0, 0.0])
    y_axis = np.array([0.0, 1.0])
    sqrt_2 = np.sqrt(2.0)

    @pytest.fixture
    def spinning_on_the_spot(self) -> xr.DataArray:
        """Simulate data for an individual's head spinning on the spot.

        The left / right keypoints move in a circular motion counter-clockwise
        around the unit circle centred on the origin, always opposite each
        other.
        The left keypoint starts on the negative x-axis, and the motion is
        split into 8 time points of uniform rotation angles.
        """
        data = np.zeros(shape=(8, 2, 2), dtype=float)
        data[:, :, 0] = np.array(
            [
                -self.x_axis,
                (-self.x_axis - self.y_axis) / self.sqrt_2,
                -self.y_axis,
                (self.x_axis - self.y_axis) / self.sqrt_2,
                self.x_axis,
                (self.x_axis + self.y_axis) / self.sqrt_2,
                self.y_axis,
                (-self.x_axis + self.y_axis) / self.sqrt_2,
            ]
        )
        data[:, :, 1] = -data[:, :, 0]
        return xr.DataArray(
            data=data,
            dims=["time", "space", "keypoint"],
            coords={"space": ["x", "y"], "keypoint": ["left", "right"]},
        )

    @pytest.mark.parametrize(
        ["swap_left_right", "swap_camera_view"],
        [
            pytest.param(True, True, id="(TT) LR, Camera"),
            pytest.param(True, False, id="(TF) LR"),
            pytest.param(False, True, id="(FT) Camera"),
            pytest.param(False, False, id="(FF)"),
        ],
    )
    def test_antisymmetry_properties(
        self,
        push_into_range,
        spinning_on_the_spot: xr.DataArray,
        swap_left_right: bool,
        swap_camera_view: bool,
    ) -> None:
        r"""Test antisymmetry arises where expected.

        Reversing the right and left keypoints, or the camera position, has the
        effect of mapping angles to the "opposite side" of the unit circle.
        Explicitly;
        - :math:`\theta <= 0` is mapped to :math:`\theta + 180`,
        - :math:`\theta > 0` is mapped to :math:`\theta - 180`.

        In theory, the antisymmetry of ``angle_rotates`` should be covered by
        the underlying tests for ``compute_signed_angle_2d``, however we
        include this case here for additional checks in conjunction with other
        behaviour.
        """
        reference_vector = self.x_axis
        left_keypoint = "left"
        right_keypoint = "right"

        args_to_function = {}
        if swap_left_right:
            args_to_function["left_keypoint"] = right_keypoint
            args_to_function["right_keypoint"] = left_keypoint
        else:
            args_to_function["left_keypoint"] = left_keypoint
            args_to_function["right_keypoint"] = right_keypoint
        if swap_camera_view:
            args_to_function["camera_view"] = "bottom_up"

        # mypy call here is angry, https://github.com/python/mypy/issues/1969
        with_orientations_swapped = kinematics.compute_forward_vector_angle(
            data=spinning_on_the_spot,
            reference_vector=reference_vector,
            **args_to_function,  # type: ignore[arg-type]
        )
        without_orientations_swapped = kinematics.compute_forward_vector_angle(
            data=spinning_on_the_spot,
            left_keypoint=left_keypoint,
            right_keypoint=right_keypoint,
            reference_vector=reference_vector,
        )
        assert without_orientations_swapped.name == "forward_vector_angle"
        assert with_orientations_swapped.name == "forward_vector_angle"

        expected_orientations = without_orientations_swapped.copy(deep=True)
        if swap_left_right:
            expected_orientations = push_into_range(
                expected_orientations + np.pi, lower=-np.pi, upper=np.pi
            )
        if swap_camera_view:
            expected_orientations = push_into_range(
                expected_orientations + np.pi, lower=-np.pi, upper=np.pi
            )
        expected_orientations = push_into_range(expected_orientations)

        xr.testing.assert_allclose(
            with_orientations_swapped, expected_orientations
        )

    def test_in_degrees_toggle(
        self, spinning_on_the_spot: xr.DataArray
    ) -> None:
        """Test that angles can be returned in degrees or radians."""
        reference_vector = self.x_axis
        left_keypoint = "left"
        right_keypoint = "right"

        in_radians = kinematics.compute_forward_vector_angle(
            data=spinning_on_the_spot,
            left_keypoint=left_keypoint,
            right_keypoint=right_keypoint,
            reference_vector=reference_vector,
            in_degrees=False,
        )
        in_degrees = kinematics.compute_forward_vector_angle(
            data=spinning_on_the_spot,
            left_keypoint=left_keypoint,
            right_keypoint=right_keypoint,
            reference_vector=reference_vector,
            in_degrees=True,
        )
        assert in_radians.name == "forward_vector_angle"
        assert in_degrees.name == "forward_vector_angle"

        xr.testing.assert_allclose(in_degrees, np.rad2deg(in_radians))

    @pytest.mark.parametrize(
        ["transformation"],
        [pytest.param("scale"), pytest.param("translation")],
    )
    def test_transformation_invariance(
        self,
        spinning_on_the_spot: xr.DataArray,
        transformation: Literal["scale", "translation"],
    ) -> None:
        """Test that certain transforms of the data have no effect on
        the relative angle computed.

        - Translations applied to both keypoints (even if the translation
        changes with time) should not affect the result, so long as both
        keypoints receive the same translation (at each timepoint).
        - Scaling the right to left keypoint vector should not produce a
        different angle.
        """
        left_keypoint = "left"
        right_keypoint = "right"
        reference_vector = self.x_axis

        translated_data = spinning_on_the_spot.values.copy()
        n_time_pts = translated_data.shape[0]

        if transformation == "translation":
            # Effectively, the data is being translated (1,1)/time-point,
            # but its keypoints are staying in the same relative positions.
            translated_data += np.arange(n_time_pts).reshape(n_time_pts, 1, 1)
        elif transformation == "scale":
            # The left keypoint position is "stretched" further away from the
            # origin over time; for the time-point at index t,
            # a scale factor of (t+1) is applied to the left keypoint.
            # The right keypoint remains unscaled, but remains in the same
            # direction away from the left keypoint.
            translated_data[:, :, 0] *= np.arange(1, n_time_pts + 1).reshape(
                n_time_pts, 1
            )
        else:
            raise ValueError(f"Did not recognise case: {transformation}")
        translated_data = spinning_on_the_spot.copy(
            deep=True, data=translated_data
        )

        untranslated_output = kinematics.compute_forward_vector_angle(
            spinning_on_the_spot,
            left_keypoint=left_keypoint,
            right_keypoint=right_keypoint,
            reference_vector=reference_vector,
        )
        translated_output = kinematics.compute_forward_vector_angle(
            spinning_on_the_spot,
            left_keypoint=left_keypoint,
            right_keypoint=right_keypoint,
            reference_vector=reference_vector,
        )

        assert untranslated_output.name == "forward_vector_angle"
        assert translated_output.name == "forward_vector_angle"

        xr.testing.assert_allclose(untranslated_output, translated_output)

    def test_casts_from_tuple(
        self, spinning_on_the_spot: xr.DataArray
    ) -> None:
        """Test that tuples and lists are cast to numpy arrays,
        when given as the reference vector.
        """
        x_axis_as_tuple = (1.0, 0.0)
        x_axis_as_list = [1.0, 0.0]

        pass_numpy = kinematics.compute_forward_vector_angle(
            spinning_on_the_spot, "left", "right", self.x_axis
        )
        pass_tuple = kinematics.compute_forward_vector_angle(
            spinning_on_the_spot, "left", "right", x_axis_as_tuple
        )
        pass_list = kinematics.compute_forward_vector_angle(
            spinning_on_the_spot, "left", "right", x_axis_as_list
        )

        xr.testing.assert_allclose(pass_numpy, pass_tuple)
        xr.testing.assert_allclose(pass_numpy, pass_list)
