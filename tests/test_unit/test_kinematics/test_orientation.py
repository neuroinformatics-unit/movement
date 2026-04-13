import re
from typing import Literal

import numpy as np
import pytest
import xarray as xr

from movement import kinematics
from movement.kinematics import compute_turning_angle


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
        for preserved_coord in ["time", "space", "individuals"]:
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
    for preserved_coord in ["time", "space", "individuals"]:
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
            dims=["time", "space", "keypoints"],
            coords={"space": ["x", "y"], "keypoints": ["left", "right"]},
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


class TestTurningAngle:
    """Test the compute_turning_angle function."""

    @pytest.mark.parametrize(
        "in_degrees, expected_units", [(True, "degrees"), (False, "radians")]
    )
    def test_output_shape_and_attributes(
        self, valid_data_array_for_forward_vector, in_degrees, expected_units
    ):
        """Test that the function returns the correct shape,
        dimensions, and attributes.
        """
        angles = compute_turning_angle(
            valid_data_array_for_forward_vector, in_degrees=in_degrees
        )

        # Space dimension must be dropped, others preserved
        assert (
            angles.sizes["time"]
            == valid_data_array_for_forward_vector.sizes["time"]
        )
        assert "space" not in angles.dims
        assert "individuals" in angles.dims

        # Attributes
        assert angles.name == "turning_angle"
        assert angles.attrs.get("units") == expected_units

    @pytest.mark.parametrize(
        "invalid_data, expected_error, expected_match_str,",
        [
            pytest.param(
                xr.DataArray(
                    np.zeros((3, 2)),
                    dims=["frame", "space"],
                    coords={"space": ["x", "y"]},
                ),
                ValueError,
                "Input data must contain ['time']",
                id="missing_time_dim",
            ),
            pytest.param(
                xr.DataArray(
                    np.zeros((3, 2)),
                    dims=["time", "axis"],
                    coords={"axis": ["x", "y"]},
                ),
                ValueError,
                "Input data must contain ['space']",
                id="missing_space_dim",
            ),
            pytest.param(
                xr.DataArray(
                    np.zeros((3, 3)),
                    dims=["time", "space"],
                    coords={"space": ["x", "y", "z"]},
                ),
                ValueError,
                "Dimension 'space' must only contain",
                id="3d_space_coords",
            ),
        ],
    )
    def test_compute_turning_angle_with_invalid_input(
        self, invalid_data, expected_error, expected_match_str
    ):
        """Test that invalid inputs raise the expected error."""
        with pytest.raises(
            expected_error, match=re.escape(expected_match_str)
        ):
            compute_turning_angle(invalid_data)

    @pytest.mark.parametrize(
        "positions, expected_angle_deg",
        [
            pytest.param(
                [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], 0.0, id="straight_line"
            ),
            pytest.param(
                [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], 90.0, id="pos_turn_90"
            ),
            pytest.param(
                [[0.0, 0.0], [1.0, 0.0], [1.0, -1.0]], -90.0, id="neg_turn_90"
            ),
            pytest.param(
                [
                    [0.0, 0.0],
                    [np.cos(np.deg2rad(170)), np.sin(np.deg2rad(170))],
                    [
                        np.cos(np.deg2rad(170)) + np.cos(np.deg2rad(-170)),
                        np.sin(np.deg2rad(170)) + np.sin(np.deg2rad(-170)),
                    ],
                ],
                20.0,
                id="wrap_across_pi_boundary",
            ),
            pytest.param(
                [[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]], 180.0, id="u_turn_180"
            ),
        ],
    )
    def test_known_turning_angles(self, positions, expected_angle_deg):
        """Test mathematical correctness of
        turning angles for specific trajectories.
        """
        pos_array = np.array(positions)
        data = xr.DataArray(
            pos_array,
            dims=["time", "space"],
            coords={"time": np.arange(len(pos_array)), "space": ["x", "y"]},
        )

        angles = compute_turning_angle(data, in_degrees=True)
        assert angles.attrs.get("units") == "degrees"
        assert np.isclose(
            angles.isel(time=2).item(), expected_angle_deg, atol=1e-6
        )

    @pytest.mark.parametrize(
        "min_step, expect_nan",
        [
            pytest.param(0.0, False, id="default_threshold"),
            pytest.param(1e-4, True, id="small_threshold"),
            pytest.param(5, True, id="large_threshold"),
        ],
    )
    def test_min_step_length_masking(self, min_step, expect_nan):
        """Test that steps smaller than min_step_length
        result in NaN turning angles.
        """
        # Trajectory with a tiny "jitter" step in the middle
        # t0 -> t1: length 1.0
        # t1 -> t2: length ~1e-5
        # t2 -> t3: length ~1.0
        positions = np.array(
            [[0.0, 0.0], [1.0, 0.0], [1.0 + 1e-5, 1e-5], [2.0, 1e-5]]
        )
        data = xr.DataArray(
            positions,
            dims=["time", "space"],
            coords={"time": np.arange(len(positions)), "space": ["x", "y"]},
        )

        angles = compute_turning_angle(data, min_step_length=min_step)

        # Verify that angles at t=2 and t=3 are NaN if the
        # min_step_length threshold is exceeded
        assert np.isnan(angles.isel(time=2).item()) == expect_nan
        assert np.isnan(angles.isel(time=3).item()) == expect_nan

    @pytest.mark.parametrize(
        "positions",
        [
            pytest.param(
                [[5.0, 5.0], [5.0, 5.0], [5.0, 5.0], [5.0, 5.0]],
                id="stationary",
            ),
            pytest.param([[0.0, 0.0], [1.0, 0.0]], id="only_two_timepoints"),
        ],
    )
    def test_all_nan_output(self, positions):
        """Test cases where all turning angles should be NaN."""
        data = xr.DataArray(
            np.array(positions),
            dims=["time", "space"],
            coords={
                "time": np.arange(len(positions)),
                "space": ["x", "y"],
            },
        )
        angles = compute_turning_angle(data)
        assert angles.isnull().all()

    def test_nan_propagation(self, valid_data_array_for_forward_vector):
        """Test that a NaN position correctly
        invalidates adjacent turning angles.
        """
        # Convert to float so we can safely insert np.nan
        data = valid_data_array_for_forward_vector.copy().astype(float)

        # Explicit, guaranteed assignment using .loc
        data.loc[
            {"time": 2, "individuals": "id_0", "keypoints": "left_ear"}
        ] = np.nan  # type: ignore[index]

        angles = compute_turning_angle(data)

        # A NaN at t=2 must break the steps at t=2 and t=3
        assert np.isnan(
            angles.sel(time=2, individuals="id_0", keypoints="left_ear").item()
        )
        assert np.isnan(
            angles.sel(time=3, individuals="id_0", keypoints="left_ear").item()
        )

    def test_stationary_keypoint_independent_masking(self):
        """Zero-step masking is per-keypoint:
        moving kp has valid angles; stationary kp all NaN.
        """
        # Create a 4-timestep array with 2 keypoints
        # Shape: (time, keypoints, space)
        data = np.zeros((4, 2, 2))

        # kp_0: moves along x-axis (0, 1, 2, 3)
        data[:, 0, 0] = [0, 1, 2, 3]
        # kp_1: stays completely stationary at (5.0, 5.0)
        data[:, 1, :] = 5.0

        ds = xr.DataArray(
            data,
            dims=["time", "keypoints", "space"],
            coords={
                "time": np.arange(4),
                "keypoints": ["kp_0", "kp_1"],
                "space": ["x", "y"],
            },
        )

        angles = compute_turning_angle(ds)

        # Moving keypoint (kp_0): NaN at t=0, t=1; valid (0.0) at t=2, t=3
        angles_kp0 = angles.isel(keypoints=0)
        assert np.isnan(angles_kp0.isel(time=0).item())
        assert np.isnan(angles_kp0.isel(time=1).item())
        assert np.allclose(
            angles_kp0.isel(time=slice(2, None)).values, 0.0, atol=1e-10
        )

        # Stationary keypoint (kp_1): should be all NaN
        angles_kp1 = angles.isel(keypoints=1)
        assert np.all(np.isnan(angles_kp1.values))
