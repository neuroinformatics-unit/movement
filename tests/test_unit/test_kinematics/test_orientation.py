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
    individual = ["id_0"]
    keypoint = ["left_ear", "right_ear", "nose"]
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
            "individual": individual,
            "keypoint": keypoint,
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


def _rotating_angle(
    angular_velocity=10.0,
    angular_acceleration=0.0,
    n_frames=200,
    fps=40,
    time_in_frames=False,
):
    """Return wrapped angles (radians) of a rotation crossing +-pi."""
    time = np.arange(n_frames) / fps
    phase = angular_velocity * time + 0.5 * angular_acceleration * time**2
    coords = np.arange(n_frames) if time_in_frames else time
    return xr.DataArray(
        np.angle(np.exp(1j * phase)), dims="time", coords={"time": coords}
    )


def _angle_to_vector(angle, scale=1.0):
    """Return 2D vectors (space=[x, y]) with the given angles."""
    return (
        xr.concat([scale * np.cos(angle), scale * np.sin(angle)], dim="space")
        .assign_coords(space=["x", "y"])
        .transpose(*angle.dims, "space")
    )


@pytest.mark.parametrize(
    "as_vector", [False, True], ids=["angle_input", "vector_input"]
)
@pytest.mark.parametrize(
    "time_in_frames, expected",
    [
        pytest.param(False, 10.0, id="time_in_seconds"),
        pytest.param(True, 10.0 / 40, id="time_in_frames"),
    ],
)
def test_angular_velocity_constant_rotation(
    as_vector, time_in_frames, expected
):
    """A constant rotation crossing +-pi yields a constant velocity."""
    angle = _rotating_angle(time_in_frames=time_in_frames)
    data = _angle_to_vector(angle, scale=3.0) if as_vector else angle
    result = kinematics.compute_angular_velocity(data)
    assert result.name == "angular_velocity"
    assert result.dims == ("time",)
    np.testing.assert_allclose(result, expected, atol=1e-9)


def test_angular_velocity_in_degrees():
    """``in_degrees=True`` converts the output to degrees."""
    result = kinematics.compute_angular_velocity(
        _rotating_angle(), in_degrees=True
    )
    np.testing.assert_allclose(result, np.rad2deg(10.0), atol=1e-6)


def test_angular_velocity_sign_convention():
    """Rotating from +x towards +y (clockwise in image coordinates)
    gives a positive angular velocity.
    """
    angle = xr.DataArray(
        [0.0, 0.1, 0.2], dims="time", coords={"time": [0, 1, 2]}
    )
    result = kinematics.compute_angular_velocity(_angle_to_vector(angle))
    assert (result > 0).all()


def test_angular_velocity_nans_stay_local():
    """NaNs only affect neighbouring frames; values after a gap stay
    correct (unlike ``np.unwrap``, which poisons the rest of the series).
    """
    angle = _rotating_angle()
    angle[[50, 51, 120]] = np.nan
    result = kinematics.compute_angular_velocity(angle)
    assert result.isnull().sum() <= 9
    np.testing.assert_allclose(result.dropna("time"), 10.0, atol=1e-9)


def test_zero_length_vector_treated_as_nan():
    """A null orientation vector is undefined and is treated as NaN."""
    vector = _angle_to_vector(_rotating_angle())
    vector[60] = 0.0
    result = kinematics.compute_angular_velocity(vector)
    np.testing.assert_allclose(result.dropna("time"), 10.0, atol=1e-9)


@pytest.mark.parametrize("window", [None, 5], ids=["raw", "window"])
def test_extra_dims_and_dim_order(window):
    """Each series is handled independently, whatever the dim order."""
    angle = _rotating_angle(angular_velocity=0.5, fps=1)
    data = xr.concat([angle, -angle], dim="individual").assign_coords(
        individual=["id_0", "id_1"]
    )
    assert data.dims == ("individual", "time")
    result = kinematics.compute_angular_velocity(data, window=window)
    assert result.dims == ("individual", "time")
    np.testing.assert_allclose(
        result.isel(time=slice(2, -2)).sel(individual="id_1"), -0.5
    )


def test_angular_time_derivative_order_2():
    """Order 2 recovers a constant angular acceleration."""
    angle = _rotating_angle(angular_velocity=1.0, angular_acceleration=4.0)
    result = kinematics.compute_angular_time_derivative(angle, order=2)
    np.testing.assert_allclose(result.isel(time=slice(2, -2)), 4.0, atol=1e-6)


@pytest.mark.parametrize(
    "data, kwargs, expected_exception",
    [
        pytest.param(
            _rotating_angle().values, {}, TypeError, id="not_a_dataarray"
        ),
        pytest.param(
            _rotating_angle().rename(time="frame"),
            {},
            ValueError,
            id="no_time_dim",
        ),
        pytest.param(
            xr.DataArray(
                np.ones((5, 3)),
                dims=["time", "space"],
                coords={"time": np.arange(5), "space": ["x", "y", "z"]},
            ),
            {},
            ValueError,
            id="3d_space",
        ),
        pytest.param(
            _rotating_angle(), {"order": 0}, ValueError, id="order_zero"
        ),
        pytest.param(
            _rotating_angle(), {"order": 1.0}, ValueError, id="order_float"
        ),
    ],
)
def test_angular_time_derivative_invalid_inputs(
    data, kwargs, expected_exception
):
    """Invalid inputs raise informative errors."""
    with pytest.raises(expected_exception):
        kinematics.compute_angular_time_derivative(
            data, **{"order": 1, **kwargs}
        )


@pytest.mark.parametrize(
    "as_vector", [False, True], ids=["angle_input", "vector_input"]
)
@pytest.mark.parametrize(
    "time_in_frames, expected",
    [
        pytest.param(False, 10.0, id="time_in_seconds"),
        pytest.param(True, 10.0 / 40, id="time_in_frames"),
    ],
)
def test_angular_velocity_window_constant_rotation(
    as_vector, time_in_frames, expected
):
    """A windowed estimate of a constant rotation is exact away from the
    edges (the first and last ``window // 2`` frames are biased).
    """
    window = 7
    angle = _rotating_angle(time_in_frames=time_in_frames)
    data = _angle_to_vector(angle) if as_vector else angle
    result = kinematics.compute_angular_velocity(data, window=window)
    h = window // 2
    np.testing.assert_allclose(
        result.isel(time=slice(h, -h)), expected, atol=1e-6
    )


def test_angular_velocity_window_matches_polyfit():
    """The windowed estimate equals the slope of a least-squares line
    over the centred window, and the trailing-window recipe equals the
    slope over the window ending at each time point.
    """
    window, h, i = 7, 3, 100
    rng = np.random.default_rng(seed=42)
    angle = _rotating_angle()
    noisy = angle.copy(
        data=np.angle(
            np.exp(1j * (angle.values + rng.normal(0, 0.05, angle.size)))
        )
    )
    t, unwrapped = noisy.time.values, np.unwrap(noisy.values)
    result = kinematics.compute_angular_velocity(noisy, window=window)
    centred_win = slice(i - h, i + h + 1)
    centred = np.polyfit(t[centred_win], unwrapped[centred_win], 1)
    assert np.isclose(result.values[i], centred[0])
    trailing_win = slice(i - window + 1, i + 1)
    trailing = np.polyfit(t[trailing_win], unwrapped[trailing_win], 1)
    assert np.isclose(result.shift(time=h).values[i], trailing[0])
    raw = kinematics.compute_angular_velocity(noisy)
    assert result.std() < raw.std()


@pytest.mark.parametrize(
    "nan_frames",
    [
        pytest.param([50, 51, 120], id="interior_gaps"),
        pytest.param([0, 1, 2, 197, 198, 199], id="leading_trailing"),
    ],
)
def test_window_with_nans(nan_frames):
    """NaNs (including at the edges) do not raise, and stay local."""
    angle = _rotating_angle()
    angle[nan_frames] = np.nan
    result = kinematics.compute_angular_velocity(angle, window=5)
    assert result.isnull().sum() <= 5 * len(nan_frames)
    interior = result.isel(time=slice(5, -5)).dropna("time")
    np.testing.assert_allclose(interior, 10.0, atol=1e-6)


def test_angular_time_derivative_order_2_window():
    """A windowed order-2 derivative recovers a constant acceleration."""
    angle = _rotating_angle(angular_velocity=1.0, angular_acceleration=4.0)
    result = kinematics.compute_angular_time_derivative(
        angle, order=2, window=9
    )
    np.testing.assert_allclose(result.isel(time=slice(5, -5)), 4.0, atol=1e-6)


def test_window_preserves_upstream_log():
    """The internal savgol_filter call does not leak into ``log``."""
    angle = _rotating_angle()
    angle.attrs["log"] = '[{"operation": "upstream"}]'
    result = kinematics.compute_angular_velocity(angle, window=5)
    assert result.attrs.get("log") == '[{"operation": "upstream"}]'


@pytest.mark.parametrize(
    "data, window",
    [
        pytest.param(_rotating_angle(), 1, id="window_too_small"),
        pytest.param(
            _rotating_angle().assign_coords(time=np.arange(200) ** 1.1),
            5,
            id="non_uniform_time",
        ),
    ],
)
def test_window_invalid_inputs(data, window):
    """Invalid windows or non-uniform time raise a ValueError."""
    with pytest.raises(ValueError):
        kinematics.compute_angular_velocity(data, window=window)
