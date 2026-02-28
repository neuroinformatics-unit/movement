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
        dims=["time", "individuals", "keypoints", "space"],
        coords={
            "time": time,
            "individuals": individuals,
            "keypoints": keypoints,
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
    """Return a position DataArray in which the ``keypoints`` dimension has
    been dropped.
    """
    return valid_data_array_for_forward_vector.sel(keypoints="nose", drop=True)


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
        | (valid_data_array_for_forward_vector.keypoints != "left_ear")
    )
    return nan_dataarray


@pytest.fixture
def valid_data_array_for_vector_from_to():
    """Return a position data array for an individual with 2 keypoints
    (neck and nose), tracked for 4 frames, in x-y space.
    """
    time = [0, 1, 2, 3]
    individuals = ["id_0"]
    keypoints = ["neck", "nose"]
    space = ["x", "y"]

    ds = xr.DataArray(
        [
            [[[0, 0], [1, 0]]],  # time 0: nose right
            [[[0, 0], [0, 1]]],  # time 1: nose down
            [[[1, 1], [0, 1]]],  # time 2: nose left
            [[[2, 0], [2, -3]]],  # time 3: nose up
        ],
        dims=["time", "individuals", "keypoints", "space"],
        coords={
            "time": time,
            "individuals": individuals,
            "keypoints": keypoints,
            "space": space,
        },
    )
    return ds


def test_compute_perpendicular_vector(
    valid_data_array_for_forward_vector,
):
    """Test that the correct output perpendicular direction vectors
    are computed from a valid mock dataset.
    """
    perp_vector = kinematics.compute_perpendicular_vector(
        valid_data_array_for_forward_vector,
        "left_ear",
        "right_ear",
        camera_view="bottom_up",
    )
    perp_vector_flipped = kinematics.compute_perpendicular_vector(
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
    assert perp_vector.name == "perpendicular_vector"
    assert perp_vector_flipped.name == "perpendicular_vector"
    assert head_vector.name == "head_direction_vector"

    known_vectors = np.array(
        [[[0, -1]], [[1, 0]], [[0, 1]], [[-1, 0]]]
    )

    for output_array in [
        perp_vector,
        perp_vector_flipped,
        head_vector,
    ]:
        assert isinstance(output_array, xr.DataArray)
        for preserved_coord in [
            "time",
            "space",
            "individuals",
        ]:
            assert np.all(
                output_array[preserved_coord]
                == valid_data_array_for_forward_vector[
                    preserved_coord
                ]
            )
        assert set(output_array["space"].values) == {"x", "y"}
    assert np.equal(perp_vector.values, known_vectors).all()
    assert np.equal(
        perp_vector_flipped.values, known_vectors * -1
    ).all()
    assert head_vector.equals(perp_vector)


def test_compute_forward_vector_deprecated(
    valid_data_array_for_forward_vector,
):
    """Test that compute_forward_vector emits a DeprecationWarning
    and returns the same result as compute_perpendicular_vector.
    """
    with pytest.warns(
        DeprecationWarning,
        match="compute_forward_vector.*deprecated",
    ):
        fwd = kinematics.compute_forward_vector(
            valid_data_array_for_forward_vector,
            "left_ear",
            "right_ear",
        )
    assert fwd.name == "forward_vector"
    perp = kinematics.compute_perpendicular_vector(
        valid_data_array_for_forward_vector,
        "left_ear",
        "right_ear",
    )
    # Values should be equal (just different names)
    assert np.allclose(fwd.values, perp.values)


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
            "Input data must contain ['keypoints']",
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
def test_compute_perpendicular_vector_with_invalid_input(
    input_data,
    keypoints,
    expected_error,
    expected_match_str,
    request,
):
    """Test that ``compute_perpendicular_vector`` catches errors
    correctly when passed invalid inputs.
    """
    # Get fixture
    input_data = request.getfixturevalue(input_data)

    # Catch error
    with pytest.raises(
        expected_error, match=re.escape(expected_match_str)
    ):
        kinematics.compute_perpendicular_vector(
            input_data, keypoints[0], keypoints[1]
        )


def test_nan_behavior_perpendicular_vector(
    valid_data_array_for_forward_vector_with_nan,
):
    """Test that ``compute_perpendicular_vector()`` generates the
    expected output for a valid input DataArray containing ``NaN``
    position values at a single time (``1``) and keypoint
    (``left_ear``).
    """
    nan_time = 1
    perp_vector = kinematics.compute_perpendicular_vector(
        valid_data_array_for_forward_vector_with_nan,
        "left_ear",
        "right_ear",
    )
    assert perp_vector.name == "perpendicular_vector"
    # Check coord preservation
    for preserved_coord in ["time", "space", "individuals"]:
        assert np.all(
            perp_vector[preserved_coord]
            == valid_data_array_for_forward_vector_with_nan[
                preserved_coord
            ]
        )
    assert set(perp_vector["space"].values) == {"x", "y"}
    # Should have NaN values at time 1
    nan_values = perp_vector.sel(time=nan_time)
    assert nan_values.shape == (1, 2)
    assert np.isnan(nan_values).all(), (
        "NaN values not returned where expected!"
    )
    # Should have no NaN values in other positions
    assert not np.isnan(
        perp_vector.sel(
            time=[
                t
                for t in (
                    valid_data_array_for_forward_vector_with_nan.time
                )
                if t != nan_time
            ]
        )
    ).any()


class TestVectorFromTo:
    """Test the compute_vector_from_to function."""

    def test_basic_output(
        self, valid_data_array_for_vector_from_to
    ):
        """Test that correct unit direction vectors are computed
        from a valid mock dataset.
        """
        result = kinematics.compute_vector_from_to(
            valid_data_array_for_vector_from_to,
            from_keypoint="neck",
            to_keypoint="nose",
        )
        assert result.name == "vector_from_to"
        assert isinstance(result, xr.DataArray)
        assert "keypoints" not in result.dims
        for coord in ["time", "space", "individuals"]:
            assert coord in result.dims

        # Check the computed unit vectors
        # time 0: [1,0] -> [1,0]
        # time 1: [0,1] -> [0,1]
        # time 2: [-1,0] -> [-1,0]
        # time 3: [0,-3] -> [0,-1]
        expected = np.array(
            [
                [[1.0, 0.0]],
                [[0.0, 1.0]],
                [[-1.0, 0.0]],
                [[0.0, -1.0]],
            ]
        )
        np.testing.assert_allclose(
            result.values, expected, atol=1e-10
        )

    def test_coord_preservation(
        self, valid_data_array_for_vector_from_to
    ):
        """Test that time, space, and individuals coords are
        preserved, but keypoints is dropped.
        """
        result = kinematics.compute_vector_from_to(
            valid_data_array_for_vector_from_to,
            from_keypoint="neck",
            to_keypoint="nose",
        )
        for coord in ["time", "space", "individuals"]:
            assert np.all(
                result[coord]
                == valid_data_array_for_vector_from_to[coord]
            )

    def test_identical_keypoints_raises(
        self, valid_data_array_for_vector_from_to
    ):
        """Test identical from/to keypoints raise ValueError."""
        with pytest.raises(
            ValueError, match="may not be identical"
        ):
            kinematics.compute_vector_from_to(
                valid_data_array_for_vector_from_to,
                from_keypoint="neck",
                to_keypoint="neck",
            )

    def test_invalid_type_raises(self):
        """Test that non-DataArray input raises TypeError."""
        with pytest.raises(
            TypeError, match="must be an xarray.DataArray"
        ):
            kinematics.compute_vector_from_to(
                np.array([1, 2, 3]),
                from_keypoint="a",
                to_keypoint="b",
            )

    def test_nan_propagation(
        self, valid_data_array_for_vector_from_to
    ):
        """Test that NaN in one keypoint propagates to output."""
        data_with_nan = (
            valid_data_array_for_vector_from_to.astype(float)
            .copy(deep=True)
        )
        data_with_nan.loc[
            {"time": 1, "keypoints": "neck"}
        ] = np.nan
        result = kinematics.compute_vector_from_to(
            data_with_nan,
            from_keypoint="neck",
            to_keypoint="nose",
        )
        assert np.isnan(result.sel(time=1)).all()
        assert not np.isnan(
            result.sel(time=[0, 2, 3])
        ).any()


class TestVectorAngle:
    """Test the compute_vector_angle function."""

    x_axis = np.array([1.0, 0.0])
    y_axis = np.array([0.0, 1.0])

    @pytest.fixture
    def simple_vectors(self) -> xr.DataArray:
        """Return vectors pointing in 4 cardinal directions."""
        data = np.array(
            [
                [1.0, 0.0],   # right (0 rad)
                [0.0, 1.0],   # down (pi/2)
                [-1.0, 0.0],  # left (pi)
                [0.0, -1.0],  # up (-pi/2)
            ]
        )
        return xr.DataArray(
            data=data,
            dims=["time", "space"],
            coords={
                "time": [0, 1, 2, 3],
                "space": ["x", "y"],
            },
        )

    def test_basic_angles(self, simple_vectors):
        """Test angles relative to x-axis for cardinal vectors."""
        angles = kinematics.compute_vector_angle(
            simple_vectors, reference_vector=self.x_axis
        )
        assert angles.name == "vector_angle"
        expected = np.array(
            [0.0, np.pi / 2, np.pi, -np.pi / 2]
        )
        np.testing.assert_allclose(
            angles.values, expected, atol=1e-10
        )

    def test_in_degrees(self, simple_vectors):
        """Test angle conversion to degrees."""
        in_rad = kinematics.compute_vector_angle(
            simple_vectors, in_degrees=False
        )
        in_deg = kinematics.compute_vector_angle(
            simple_vectors, in_degrees=True
        )
        assert in_rad.name == "vector_angle"
        assert in_deg.name == "vector_angle"
        xr.testing.assert_allclose(
            in_deg, np.rad2deg(in_rad)
        )

    def test_tuple_and_list_reference(self, simple_vectors):
        """Test that tuple/list references are cast properly."""
        from_np = kinematics.compute_vector_angle(
            simple_vectors, reference_vector=self.x_axis
        )
        from_tuple = kinematics.compute_vector_angle(
            simple_vectors, reference_vector=(1.0, 0.0)
        )
        from_list = kinematics.compute_vector_angle(
            simple_vectors, reference_vector=[1.0, 0.0]
        )
        xr.testing.assert_allclose(from_np, from_tuple)
        xr.testing.assert_allclose(from_np, from_list)

    def test_time_varying_reference(self, simple_vectors):
        """Test with a time-varying reference vector."""
        ref = xr.DataArray(
            data=np.array(
                [
                    [1.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [0.0, 1.0],
                ]
            ),
            dims=["time", "space"],
            coords={
                "time": [0, 1, 2, 3],
                "space": ["x", "y"],
            },
        )
        angles = kinematics.compute_vector_angle(
            simple_vectors, reference_vector=ref
        )
        # time 0: vector [1,0], ref [1,0] -> 0
        # time 1: vector [0,1], ref [1,0] -> pi/2
        # time 2: vector [-1,0], ref [0,1] -> pi/2
        # time 3: vector [0,-1], ref [0,1] -> pi
        expected = np.array(
            [0.0, np.pi / 2, np.pi / 2, np.pi]
        )
        np.testing.assert_allclose(
            angles.values, expected, atol=1e-10
        )

    def test_invalid_input_type(self):
        """Test that non-DataArray input raises TypeError."""
        with pytest.raises(
            TypeError, match="must be an xarray.DataArray"
        ):
            kinematics.compute_vector_angle(
                np.array([1.0, 0.0])
            )


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
            dims=["time", "space", "keypoints"],
            coords={"space": ["x", "y"], "keypoints": ["left", "right"]},
        )

    @pytest.mark.filterwarnings(
        "ignore:.*deprecated:DeprecationWarning"
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

    @pytest.mark.filterwarnings(
        "ignore:.*deprecated:DeprecationWarning"
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

    @pytest.mark.filterwarnings(
        "ignore:.*deprecated:DeprecationWarning"
    )
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

    @pytest.mark.filterwarnings(
        "ignore:.*deprecated:DeprecationWarning"
    )
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

    def test_deprecated_forward_vector_angle(
        self, spinning_on_the_spot: xr.DataArray
    ) -> None:
        """Test that compute_forward_vector_angle emits a
        DeprecationWarning.
        """
        with pytest.warns(
            DeprecationWarning,
            match="compute_forward_vector_angle.*deprecated",
        ):
            kinematics.compute_forward_vector_angle(
                spinning_on_the_spot, "left", "right"
            )
