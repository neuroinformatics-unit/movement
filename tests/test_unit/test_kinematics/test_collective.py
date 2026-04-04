# test_collective.py
"""Tests for the collective behavior metrics module."""

import numpy as np
import pytest
import xarray as xr

from movement import kinematics


def _get_space_labels(n_space: int, space: list[str] | None) -> list[str]:
    """Return space labels, defaulting to ['x', 'y'] for 2D."""
    if space is not None:
        return space
    if n_space == 2:
        return ["x", "y"]
    raise ValueError("Provide explicit `space` labels for non-2D data.")


def _build_coords(
    data: np.ndarray,
    time: list | None,
    space: list[str] | None,
    individuals: list | None,
    keypoints: list[str] | None = None,
) -> dict:
    """Build coordinate dict for a position DataArray."""
    n_time, n_space = data.shape[0], data.shape[1]
    coords: dict = {
        "time": time if time is not None else list(range(n_time)),
        "space": _get_space_labels(n_space, space),
    }
    if data.ndim == 4:
        n_keypoints = data.shape[2]
        coords["keypoints"] = keypoints or [
            f"kp_{i}" for i in range(n_keypoints)
        ]
        n_individuals = data.shape[3]
    else:
        n_individuals = data.shape[2]
    coords["individuals"] = individuals or [
        f"id_{i}" for i in range(n_individuals)
    ]
    return coords


def _make_position_dataarray(
    data: np.ndarray,
    *,
    time: list | None = None,
    individuals: list | None = None,
    keypoints: list[str] | None = None,
    space: list[str] | None = None,
) -> xr.DataArray:
    """Create a position DataArray for tests."""
    data = np.asarray(data, dtype=float)

    dims_map = {
        3: ["time", "space", "individuals"],
        4: ["time", "space", "keypoints", "individuals"],
    }
    if data.ndim not in dims_map:
        raise ValueError(
            "Expected data with shape (time, space, individuals) or "
            "(time, space, keypoints, individuals)."
        )

    return xr.DataArray(
        data,
        dims=dims_map[data.ndim],
        coords=_build_coords(data, time, space, individuals, keypoints),
        name="position",
    )


@pytest.fixture
def aligned_positions() -> xr.DataArray:
    """Two individuals moving together in +x direction."""
    data = np.array(
        [
            [[0, 5], [0, 0]],
            [[1, 6], [0, 0]],
            [[2, 7], [0, 0]],
            [[3, 8], [0, 0]],
        ],
        dtype=float,
    )
    return _make_position_dataarray(data)


@pytest.fixture
def opposite_positions() -> xr.DataArray:
    """Two individuals moving in opposite x directions (+x and -x)."""
    data = np.array(
        [
            [[0, 5], [0, 0]],
            [[1, 4], [0, 0]],
            [[2, 3], [0, 0]],
            [[3, 2], [0, 0]],
        ],
        dtype=float,
    )
    return _make_position_dataarray(data)


@pytest.fixture
def partial_alignment_positions() -> xr.DataArray:
    """Three individuals: two move +x, one moves +y."""
    data = np.array(
        [
            [[0, 5, 0], [0, 0, 0]],
            [[1, 6, 0], [0, 0, 1]],
            [[2, 7, 0], [0, 0, 2]],
            [[3, 8, 0], [0, 0, 3]],
        ],
        dtype=float,
    )
    return _make_position_dataarray(data)


@pytest.fixture
def perpendicular_positions() -> xr.DataArray:
    """Four individuals moving in cardinal directions (+x, -x, +y, -y)."""
    data = np.array(
        [
            [[0, 10, 0, 0], [0, 0, 0, 10]],
            [[1, 9, 0, 0], [0, 0, 1, 9]],
            [[2, 8, 0, 0], [0, 0, 2, 8]],
            [[3, 7, 0, 0], [0, 0, 3, 7]],
        ],
        dtype=float,
    )
    return _make_position_dataarray(data)


@pytest.fixture
def keypoint_positions() -> xr.DataArray:
    """Two individuals with tail_base/neck keypoints, both facing +x."""
    data = np.array(
        [
            [
                [[0.0, 10.0], [1.0, 11.0]],
                [[0.0, 0.0], [0.0, 0.0]],
            ],
            [
                [[0.5, 10.5], [1.5, 11.5]],
                [[0.0, 0.0], [0.0, 0.0]],
            ],
            [
                [[1.0, 11.0], [2.0, 12.0]],
                [[0.0, 0.0], [0.0, 0.0]],
            ],
        ],
        dtype=float,
    )
    return _make_position_dataarray(data, keypoints=["tail_base", "neck"])


class TestComputePolarizationValidation:
    """Tests for input validation in compute_polarization."""

    def test_requires_dataarray(self):
        """Raise TypeError if input is not an xarray.DataArray."""
        with pytest.raises(TypeError, match="xarray.DataArray"):
            kinematics.compute_polarization(np.zeros((3, 2, 2)))

    @pytest.mark.parametrize(
        "dims",
        [
            ("space", "individuals"),
            ("time", "individuals"),
            ("time", "space"),
        ],
        ids=["missing_time", "missing_space", "missing_individuals"],
    )
    def test_requires_time_space_individuals(self, dims):
        """Raise ValueError if required dimensions are missing."""
        data = xr.DataArray(np.zeros((2, 2)), dims=dims)
        with pytest.raises(ValueError, match="time|space|individuals"):
            kinematics.compute_polarization(data)

    def test_rejects_unexpected_dimensions(self):
        """Raise ValueError if data contains unsupported dimensions."""
        data = xr.DataArray(
            np.zeros((3, 2, 2, 2)),
            dims=["time", "space", "individuals", "batch"],
            coords={
                "time": [0, 1, 2],
                "space": ["x", "y"],
                "individuals": ["a", "b"],
                "batch": [0, 1],
            },
        )
        with pytest.raises(ValueError, match="unsupported dimension"):
            kinematics.compute_polarization(data)

    def test_requires_x_and_y_space_labels(self):
        """Raise ValueError if space dimension lacks x and y labels."""
        data = xr.DataArray(
            np.zeros((3, 2, 2)),
            dims=["time", "space", "individuals"],
            coords={
                "time": [0, 1, 2],
                "space": ["lat", "lon"],
                "individuals": ["a", "b"],
            },
        )
        with pytest.raises(
            ValueError, match="include coordinate labels 'x' and 'y'"
        ):
            kinematics.compute_polarization(data)

    @pytest.mark.parametrize(
        "body_axis_keypoints",
        [
            "neck",
            ("tail_base",),
            ("tail_base", "neck", "ear"),
            123,
        ],
        ids=["string", "length_one", "length_three", "non_iterable"],
    )
    def test_body_axis_keypoints_must_be_length_two_iterable(
        self,
        body_axis_keypoints,
        keypoint_positions,
    ):
        """Raise TypeError if body_axis_keypoints is not length-two."""
        with pytest.raises(TypeError, match="exactly two keypoint names"):
            kinematics.compute_polarization(
                keypoint_positions,
                body_axis_keypoints=body_axis_keypoints,
            )

    def test_body_axis_keypoints_must_be_hashable(self, keypoint_positions):
        """Raise TypeError if body axis keypoints are not hashable."""
        with pytest.raises(TypeError, match="hashable"):
            kinematics.compute_polarization(
                keypoint_positions,
                body_axis_keypoints=(["tail_base"], "neck"),
            )

    def test_body_axis_keypoints_require_keypoints_dimension(
        self, aligned_positions
    ):
        """Raise ValueError if body_axis_keypoints given without keypoints."""
        with pytest.raises(
            ValueError, match="requires a 'keypoints' dimension"
        ):
            kinematics.compute_polarization(
                aligned_positions,
                body_axis_keypoints=("tail_base", "neck"),
            )

    def test_body_axis_keypoints_must_exist(self, keypoint_positions):
        """Raise ValueError if specified keypoints do not exist in data."""
        with pytest.raises(ValueError, match="snout|keypoints"):
            kinematics.compute_polarization(
                keypoint_positions,
                body_axis_keypoints=("tail_base", "snout"),
            )

    def test_body_axis_keypoints_must_be_distinct(self, keypoint_positions):
        """Raise ValueError if origin and target keypoints are identical."""
        with pytest.raises(ValueError, match="two distinct keypoint names"):
            kinematics.compute_polarization(
                keypoint_positions,
                body_axis_keypoints=("tail_base", "tail_base"),
            )

    @pytest.mark.parametrize(
        ("displacement_frames", "expected_exception"),
        [
            (0, ValueError),
            (-1, ValueError),
            (1.5, TypeError),
            (True, TypeError),
        ],
        ids=["zero", "negative", "float", "bool"],
    )
    def test_displacement_frames_must_be_positive_integer(
        self,
        aligned_positions,
        displacement_frames,
        expected_exception,
    ):
        """Raise error if displacement_frames is not a positive integer."""
        with pytest.raises(expected_exception, match="positive integer|>= 1"):
            kinematics.compute_polarization(
                aligned_positions,
                displacement_frames=displacement_frames,
            )

    def test_invalid_displacement_frames_is_ignored_in_keypoint_mode(
        self,
        keypoint_positions,
    ):
        """Invalid displacement_frames is ignored when keypoints are used."""
        polarization = kinematics.compute_polarization(
            keypoint_positions,
            body_axis_keypoints=("tail_base", "neck"),
            displacement_frames=0,
        )
        assert np.allclose(polarization.values, 1.0, atol=1e-10)

    def test_requires_space_coordinate_labels_to_exist(self):
        """Raise ValueError if the space dimension has no coordinate labels."""
        data = xr.DataArray(
            np.zeros((3, 2, 2)),
            dims=["time", "space", "individuals"],
            coords={
                "time": [0, 1, 2],
                "individuals": ["a", "b"],
            },
            name="position",
        )
        with pytest.raises(
            ValueError,
            match="coordinate labels for the 'space' dimension",
        ):
            kinematics.compute_polarization(data)

    def test_empty_keypoints_dimension_raises_in_displacement_mode(self):
        """Raise if keypoints dimension exists but contains no entries."""
        data = xr.DataArray(
            np.empty((3, 2, 0, 2)),
            dims=["time", "space", "keypoints", "individuals"],
            coords={
                "time": [0, 1, 2],
                "space": ["x", "y"],
                "keypoints": [],
                "individuals": ["a", "b"],
            },
            name="position",
        )
        with pytest.raises(ValueError, match="at least one keypoint"):
            kinematics.compute_polarization(data)


class TestComputePolarizationBehavior:
    """Tests for polarization computation behavior."""

    def test_aligned_motion_gives_one(self, aligned_positions):
        """Polarization is 1.0 when all individuals move in same direction."""
        polarization = kinematics.compute_polarization(aligned_positions)
        assert np.isnan(polarization.values[0])
        assert np.allclose(polarization.values[1:], 1.0, atol=1e-10)

    def test_opposite_motion_gives_zero(self, opposite_positions):
        """Polarization is 0.0 when individuals move in opposite directions."""
        polarization = kinematics.compute_polarization(opposite_positions)
        assert np.allclose(polarization.values[1:], 0.0, atol=1e-10)

    def test_perpendicular_cardinal_directions_give_zero(
        self, perpendicular_positions
    ):
        """Polarization is 0.0 when four individuals move in cardinal dirs."""
        polarization = kinematics.compute_polarization(perpendicular_positions)
        assert np.allclose(polarization.values[1:], 0.0, atol=1e-10)

    def test_partial_alignment_matches_expected_magnitude(
        self,
        partial_alignment_positions,
    ):
        """Polarization matches expected value for partial alignment."""
        polarization = kinematics.compute_polarization(
            partial_alignment_positions
        )
        expected = np.sqrt(5) / 3
        assert np.allclose(polarization.values[1:], expected, atol=1e-10)

    def test_single_individual_gives_one(self):
        """Polarization is 1.0 for a single moving individual."""
        data = np.array(
            [
                [[0], [0]],
                [[1], [0]],
                [[2], [0]],
                [[3], [0]],
            ],
            dtype=float,
        )
        polarization = kinematics.compute_polarization(
            _make_position_dataarray(data)
        )
        assert np.allclose(polarization.values[1:], 1.0, atol=1e-10)

    def test_stationary_individuals_are_excluded(self):
        """Stationary individuals produce NaN polarization and angle."""
        data = np.array(
            [
                [[0, 10], [0, 0]],
                [[0, 10], [0, 0]],
                [[0, 10], [0, 0]],
            ],
            dtype=float,
        )
        polarization, mean_angle = kinematics.compute_polarization(
            _make_position_dataarray(data),
            return_angle=True,
        )
        assert np.all(np.isnan(polarization.values))
        assert np.all(np.isnan(mean_angle.values))

    def test_stationary_and_moving_individuals_uses_only_valid_headings(self):
        """Only moving individuals contribute to polarization."""
        data = np.array(
            [
                [[0, 10], [0, 0]],
                [[1, 10], [0, 0]],
                [[2, 10], [0, 0]],
                [[3, 10], [0, 0]],
            ],
            dtype=float,
        )
        polarization, mean_angle = kinematics.compute_polarization(
            _make_position_dataarray(data),
            return_angle=True,
        )
        assert np.allclose(polarization.values[1:], 1.0, atol=1e-10)
        assert np.allclose(mean_angle.values[1:], 0.0, atol=1e-10)

    def test_one_coordinate_nan_excludes_that_individual(self):
        """NaN in one coordinate excludes that individual from calculation."""
        data = np.array(
            [
                [[0, 10], [0, 0]],
                [[1, np.nan], [0, 0]],
                [[2, 12], [0, 0]],
            ],
            dtype=float,
        )
        polarization, mean_angle = kinematics.compute_polarization(
            _make_position_dataarray(data),
            return_angle=True,
        )
        assert np.isnan(polarization.values[0])
        assert np.allclose(polarization.values[1:], 1.0, atol=1e-10)
        assert np.allclose(mean_angle.values[1:], 0.0, atol=1e-10)

    def test_nan_in_body_axis_heading_excludes_that_individual(self):
        """NaN in keypoint position excludes that individual."""
        data = np.array(
            [
                [
                    [[0.0, 10.0], [1.0, 11.0]],
                    [[0.0, 0.0], [0.0, 0.0]],
                ],
                [
                    [[1.0, 10.0], [2.0, np.nan]],
                    [[0.0, 0.0], [0.0, np.nan]],
                ],
                [
                    [[2.0, 12.0], [3.0, 13.0]],
                    [[0.0, 0.0], [0.0, 0.0]],
                ],
            ],
            dtype=float,
        )
        da = _make_position_dataarray(data, keypoints=["tail_base", "neck"])
        polarization = kinematics.compute_polarization(
            da,
            body_axis_keypoints=("tail_base", "neck"),
        )
        assert np.allclose(polarization.values[[0, 2]], 1.0, atol=1e-10)
        assert np.allclose(polarization.values[1], 1.0, atol=1e-10)

    def test_empty_individual_axis_returns_all_nan(self):
        """Empty individuals axis returns all NaN values."""
        data = _make_position_dataarray(
            np.empty((3, 2, 0)),
            individuals=[],
            space=["x", "y"],
        )
        polarization, mean_angle = kinematics.compute_polarization(
            data,
            return_angle=True,
        )
        assert np.all(np.isnan(polarization.values))
        assert np.all(np.isnan(mean_angle.values))

    def test_empty_time_axis_returns_empty_outputs(self):
        """Empty time axis returns empty output arrays."""
        data = xr.DataArray(
            np.empty((0, 2, 0)),
            dims=["time", "space", "individuals"],
            coords={"time": [], "space": ["x", "y"], "individuals": []},
            name="position",
        )
        polarization, mean_angle = kinematics.compute_polarization(
            data,
            return_angle=True,
        )
        assert polarization.shape == (0,)
        assert mean_angle.shape == (0,)
        assert polarization.name == "polarization"
        assert mean_angle.name == "mean_angle"

    def test_preserves_non_uniform_time_coordinates(self, aligned_positions):
        """Non-uniform time coordinates are preserved in output."""
        time = [0.0, 0.25, 0.75, 1.5]
        data = aligned_positions.assign_coords(time=time)
        polarization, mean_angle = kinematics.compute_polarization(
            data,
            return_angle=True,
        )
        np.testing.assert_array_equal(polarization.time.values, time)
        np.testing.assert_array_equal(mean_angle.time.values, time)

    def test_polarization_is_invariant_to_individual_order(self):
        """Polarization is independent of individual ordering."""
        data = np.array(
            [
                [[0, 5, 0], [0, 0, 0]],
                [[1, 6, 0], [0, 0, 1]],
                [[2, 7, 0], [0, 0, 2]],
                [[3, 8, 0], [0, 0, 3]],
            ],
            dtype=float,
        )
        da = _make_position_dataarray(data)
        da_permuted = da.isel(individuals=[2, 0, 1])

        pol_original = kinematics.compute_polarization(da)
        pol_permuted = kinematics.compute_polarization(da_permuted)

        np.testing.assert_allclose(
            pol_original.values, pol_permuted.values, atol=1e-10
        )

    def test_zero_length_body_axis_vectors_are_excluded(self):
        """Zero-length body-axis headings are excluded as invalid."""
        # ind0 has coincident tail_base and neck (zero-length heading)
        # ind1 has valid +x body axis heading
        data = np.array(
            [
                [
                    [[0.0, 10.0], [0.0, 11.0]],  # x: ind0 zero-length, ind1 +1
                    [[0.0, 0.0], [0.0, 0.0]],  # y
                ],
                [
                    [[0.0, 10.5], [0.0, 11.5]],
                    [[0.0, 0.0], [0.0, 0.0]],
                ],
            ],
            dtype=float,
        )
        da = _make_position_dataarray(data, keypoints=["tail_base", "neck"])

        polarization, mean_angle = kinematics.compute_polarization(
            da,
            body_axis_keypoints=("tail_base", "neck"),
            return_angle=True,
        )

        assert np.allclose(polarization.values, 1.0, atol=1e-10)
        assert np.allclose(mean_angle.values, 0.0, atol=1e-10)

    def test_polarization_is_invariant_to_translation(
        self,
        partial_alignment_positions,
    ):
        """Adding a constant offset does not change polarization."""
        shifted = partial_alignment_positions.copy()
        shifted.loc[{"space": "x"}] = shifted.sel(space="x") + 1000.0
        shifted.loc[{"space": "y"}] = shifted.sel(space="y") - 500.0

        pol_original = kinematics.compute_polarization(
            partial_alignment_positions
        )
        pol_shifted = kinematics.compute_polarization(shifted)

        np.testing.assert_allclose(
            pol_original.values,
            pol_shifted.values,
            atol=1e-10,
            equal_nan=True,
        )

    def test_polarization_is_invariant_to_positive_scaling(
        self,
        partial_alignment_positions,
    ):
        """Positive scalar multiplication preserves polarization."""
        scaled = partial_alignment_positions * 7.5

        pol_original = kinematics.compute_polarization(
            partial_alignment_positions
        )
        pol_scaled = kinematics.compute_polarization(scaled)

        np.testing.assert_allclose(
            pol_original.values,
            pol_scaled.values,
            atol=1e-10,
            equal_nan=True,
        )

    def test_polarization_is_invariant_to_global_rotation(
        self,
        partial_alignment_positions,
    ):
        """A global planar rotation preserves polarization magnitude."""
        x = partial_alignment_positions.sel(space="x")
        y = partial_alignment_positions.sel(space="y")

        rotated = partial_alignment_positions.copy()
        rotated.loc[{"space": "x"}] = -y
        rotated.loc[{"space": "y"}] = x

        pol_original = kinematics.compute_polarization(
            partial_alignment_positions
        )
        pol_rotated = kinematics.compute_polarization(rotated)

        np.testing.assert_allclose(
            pol_original.values,
            pol_rotated.values,
            atol=1e-10,
            equal_nan=True,
        )

    def _body_axis_baseline(self):
        """Build body-axis test data and return (da, pol, angle).

        Three individuals with body axes: +x, +x, +y.
        Vector sum = (2, 1), polarization = sqrt(5)/3, angle = atan2(1,2).
        Absolute positions differ across frames to test body-axis
        independence from location.
        """
        data = np.array(
            [
                [
                    [[0.0, 10.0, -2.0], [1.0, 11.0, -2.0]],
                    [[0.0, 5.0, 3.0], [0.0, 5.0, 4.0]],
                ],
                [
                    [[100.0, 50.0, 7.0], [101.0, 51.0, 7.0]],
                    [[-1.0, 20.0, -3.0], [-1.0, 20.0, -2.0]],
                ],
            ],
            dtype=float,
        )
        da = _make_position_dataarray(data, keypoints=["tail_base", "neck"])
        pol, angle = kinematics.compute_polarization(
            da,
            body_axis_keypoints=("tail_base", "neck"),
            return_angle=True,
        )
        return da, pol, angle

    def test_body_axis_baseline_matches_expected_values(self):
        """Body-axis polarization and angle match hand-computed values."""
        _da, pol, angle = self._body_axis_baseline()
        np.testing.assert_allclose(pol.values, np.sqrt(5) / 3, atol=1e-10)
        np.testing.assert_allclose(
            angle.values, np.arctan2(1.0, 2.0), atol=1e-10
        )

    def test_body_axis_invariance_to_translation(self):
        """Global translation does not change body-axis polarization."""
        da, pol_base, angle_base = self._body_axis_baseline()
        translated = da.copy()
        translated.loc[{"space": "x"}] = translated.sel(space="x") + 123.4
        translated.loc[{"space": "y"}] = translated.sel(space="y") - 56.7

        pol, angle = kinematics.compute_polarization(
            translated,
            body_axis_keypoints=("tail_base", "neck"),
            return_angle=True,
        )
        np.testing.assert_allclose(pol.values, pol_base.values, atol=1e-10)
        np.testing.assert_allclose(angle.values, angle_base.values, atol=1e-10)

    def test_body_axis_invariance_to_positive_scaling(self):
        """Positive scaling preserves body-axis polarization and angle."""
        da, pol_base, angle_base = self._body_axis_baseline()

        pol, angle = kinematics.compute_polarization(
            da * 4.2,
            body_axis_keypoints=("tail_base", "neck"),
            return_angle=True,
        )
        np.testing.assert_allclose(pol.values, pol_base.values, atol=1e-10)
        np.testing.assert_allclose(angle.values, angle_base.values, atol=1e-10)

    def test_body_axis_angle_rotates_under_global_rotation(self):
        """Polarization preserved, angle shifts by pi/2.

        Tests behavior under 90-degree rotation.
        """
        da, pol_base, angle_base = self._body_axis_baseline()
        rotated = da.copy()
        x = da.sel(space="x")
        y = da.sel(space="y")
        rotated.loc[{"space": "x"}] = -y
        rotated.loc[{"space": "y"}] = x

        pol, angle = kinematics.compute_polarization(
            rotated,
            body_axis_keypoints=("tail_base", "neck"),
            return_angle=True,
        )
        np.testing.assert_allclose(pol.values, pol_base.values, atol=1e-10)

        expected = angle_base.values + (np.pi / 2)
        expected = (expected + np.pi) % (2 * np.pi) - np.pi
        np.testing.assert_allclose(angle.values, expected, atol=1e-10)


class TestHeadingSourceSelection:
    """Tests for heading computation mode selection."""

    def test_body_axis_heading_valid_on_first_frame_returns_expected_angle(
        self, keypoint_positions
    ):
        """Body-axis heading is valid from frame 0 and returns angle 0."""
        polarization, mean_angle = kinematics.compute_polarization(
            keypoint_positions,
            body_axis_keypoints=("tail_base", "neck"),
            return_angle=True,
        )
        assert np.allclose(polarization.values, 1.0, atol=1e-10)
        assert np.allclose(mean_angle.values, 0.0, atol=1e-10)

    def test_displacement_mode_with_keypoints_uses_first_keypoint(self):
        """Displacement mode uses first keypoint when multiple exist."""
        data = np.array(
            [
                [
                    [[0, 10], [0, 10]],
                    [[0, 0], [0, 0]],
                ],
                [
                    [[1, 11], [1, 9]],
                    [[0, 0], [0, 0]],
                ],
                [
                    [[2, 12], [2, 8]],
                    [[0, 0], [0, 0]],
                ],
            ],
            dtype=float,
        )
        da = _make_position_dataarray(data, keypoints=["thorax", "head"])
        polarization = kinematics.compute_polarization(da)
        assert np.allclose(polarization.values[1:], 1.0, atol=1e-10)

    def test_explicit_keypoint_selection_with_sel(self):
        """Pre-selecting keypoint with .sel() uses that keypoint.

        Data shape: (time, space, keypoints, individuals).

        X-coordinates across frames:

            Keypoint | Individual | Frame 0 | Frame 1 | Displacement
            ---------|------------|---------|---------|-------------
            thorax   | ind0       | 0       | 1       | +1 (right)
            thorax   | ind1       | 10      | 11      | +1 (right)
            head     | ind0       | 0       | 1       | +1 (right)
            head     | ind1       | 10      | 9       | -1 (left)

        Thorax: both individuals move right -> polarization = 1.0
        Head: ind0 moves right, ind1 moves left -> polarization = 0.0
        """
        data = np.array(
            [
                [
                    [[0, 10], [0, 10]],
                    [[0, 0], [0, 0]],
                ],
                [
                    [[1, 11], [1, 9]],
                    [[0, 0], [0, 0]],
                ],
            ],
            dtype=float,
        )
        da = _make_position_dataarray(data, keypoints=["thorax", "head"])

        # Without .sel(): uses thorax -> both move right -> polarization = 1.0
        pol_default = kinematics.compute_polarization(da)
        assert np.allclose(pol_default.values[1], 1.0, atol=1e-10)

        # With .sel(): head selected -> ind0 right, ind1 left -> 0.0
        pol_head = kinematics.compute_polarization(da.sel(keypoints="head"))
        assert np.allclose(pol_head.values[1], 0.0, atol=1e-10)

    def test_body_axis_heading_overrides_displacement_behavior(self):
        """Body-axis heading overrides displacement computation."""
        data = np.array(
            [
                [
                    [[0.0, 0.0], [1.0, 1.0]],
                    [[0.0, 2.0], [0.0, 2.0]],
                ],
                [
                    [[0.0, 0.0], [1.0, 1.0]],
                    [[1.0, 3.0], [1.0, 3.0]],
                ],
            ],
            dtype=float,
        )
        da = _make_position_dataarray(data, keypoints=["tail_base", "neck"])
        polarization = kinematics.compute_polarization(
            da,
            body_axis_keypoints=("tail_base", "neck"),
            displacement_frames=1000,
        )
        assert np.allclose(polarization.values, 1.0, atol=1e-10)

    def test_extra_spatial_dimensions_are_ignored_for_planar_metrics(self):
        """Extra spatial dimensions (z) are ignored; only x/y used."""
        data = np.array(
            [
                [[0, 5], [0, 0], [0, 100]],
                [[1, 6], [0, 0], [10, -100]],
                [[2, 7], [0, 0], [-10, 50]],
                [[3, 8], [0, 0], [999, -999]],
            ],
            dtype=float,
        )
        da = _make_position_dataarray(data, space=["x", "y", "z"])
        polarization, mean_angle = kinematics.compute_polarization(
            da,
            return_angle=True,
        )
        assert np.allclose(polarization.values[1:], 1.0, atol=1e-10)
        assert np.allclose(mean_angle.values[1:], 0.0, atol=1e-10)


class TestDisplacementFrames:
    """Tests for displacement_frames parameter behavior."""

    def test_first_n_frames_are_nan(self, aligned_positions):
        """First N frames are NaN when displacement_frames=N."""
        polarization, mean_angle = kinematics.compute_polarization(
            aligned_positions,
            displacement_frames=2,
            return_angle=True,
        )
        assert np.all(np.isnan(polarization.values[:2]))
        assert np.all(np.isnan(mean_angle.values[:2]))
        assert np.allclose(polarization.values[2:], 1.0, atol=1e-10)
        assert np.allclose(mean_angle.values[2:], 0.0, atol=1e-10)

    def test_nan_in_reference_frame_propagates_to_that_displacement_window(
        self,
    ):
        """NaN in reference frame propagates through displacement window."""
        data = np.array(
            [
                [[0, 5], [0, 0]],
                [[np.nan, np.nan], [np.nan, np.nan]],
                [[2, 7], [0, 0]],
                [[3, 8], [0, 0]],
                [[4, 9], [0, 0]],
            ],
            dtype=float,
        )
        polarization = kinematics.compute_polarization(
            _make_position_dataarray(data),
            displacement_frames=2,
        )
        assert np.isnan(polarization.values[0])
        assert np.isnan(polarization.values[1])
        assert np.allclose(polarization.values[2], 1.0, atol=1e-10)
        assert np.isnan(polarization.values[3])
        assert np.allclose(polarization.values[4], 1.0, atol=1e-10)

    def test_larger_displacement_window_can_change_alignment_estimate(self):
        """Larger displacement window smooths jittery movement."""
        data = np.array(
            [
                [[0, 10], [0, 0]],
                [[2, 9], [0, 0]],
                [[1, 11], [0, 0]],
                [[3, 10], [0, 0]],
                [[2, 12], [0, 0]],
                [[4, 11], [0, 0]],
            ],
            dtype=float,
        )
        da = _make_position_dataarray(data)

        pol_1frame = kinematics.compute_polarization(da, displacement_frames=1)
        pol_2frame = kinematics.compute_polarization(da, displacement_frames=2)

        assert np.allclose(pol_1frame.values[1:], 0.0, atol=1e-10)
        assert np.allclose(pol_2frame.values[2:], 1.0, atol=1e-10)

    def test_displacement_frames_larger_than_time_axis_returns_all_nan(
        self,
        aligned_positions,
    ):
        """Oversized displacement windows produce no valid headings."""
        polarization, mean_angle = kinematics.compute_polarization(
            aligned_positions,
            displacement_frames=10,
            return_angle=True,
        )
        assert np.all(np.isnan(polarization.values))
        assert np.all(np.isnan(mean_angle.values))


class TestReturnAngle:
    """Tests for return_angle parameter behavior."""

    def test_default_returns_only_polarization(self, aligned_positions):
        """Default return is a single polarization DataArray."""
        result = kinematics.compute_polarization(aligned_positions)
        assert isinstance(result, xr.DataArray)
        assert result.name == "polarization"
        assert result.dims == ("time",)

    def test_return_angle_true_returns_named_pair(self, aligned_positions):
        """return_angle=True returns (polarization, mean_angle) tuple."""
        polarization, mean_angle = kinematics.compute_polarization(
            aligned_positions,
            return_angle=True,
        )
        assert isinstance(polarization, xr.DataArray)
        assert isinstance(mean_angle, xr.DataArray)
        assert (polarization.name, mean_angle.name) == (
            "polarization",
            "mean_angle",
        )
        assert polarization.dims == ("time",)
        assert mean_angle.dims == ("time",)

    @pytest.mark.parametrize(
        ("data", "expected_angle", "use_abs"),
        [
            (
                np.array(
                    [
                        [[0, 5], [0, 0]],
                        [[1, 6], [0, 0]],
                        [[2, 7], [0, 0]],
                    ],
                    dtype=float,
                ),
                0.0,
                False,
            ),
            (
                np.array(
                    [
                        [[0, 0], [0, 5]],
                        [[0, 0], [1, 6]],
                        [[0, 0], [2, 7]],
                    ],
                    dtype=float,
                ),
                np.pi / 2,
                False,
            ),
            (
                np.array(
                    [
                        [[10, 15], [0, 0]],
                        [[9, 14], [0, 0]],
                        [[8, 13], [0, 0]],
                    ],
                    dtype=float,
                ),
                np.pi,
                True,
            ),
        ],
        ids=["positive_x", "positive_y", "negative_x"],
    )
    def test_mean_angle_matches_cardinal_directions(
        self,
        data,
        expected_angle,
        use_abs,
    ):
        """Mean angle matches expected value for cardinal directions."""
        _, mean_angle = kinematics.compute_polarization(
            _make_position_dataarray(data),
            return_angle=True,
        )
        values = mean_angle.values[1:]
        if use_abs:
            values = np.abs(values)
        assert np.allclose(values, expected_angle, atol=1e-10)

    def test_mean_angle_diagonal_motion_is_pi_over_four(self):
        """Mean angle is pi/4 for diagonal (+x, +y) motion."""
        data = np.array(
            [
                [[0, 5], [0, 5]],
                [[1, 6], [1, 6]],
                [[2, 7], [2, 7]],
            ],
            dtype=float,
        )
        _, mean_angle = kinematics.compute_polarization(
            _make_position_dataarray(data),
            return_angle=True,
        )
        assert np.allclose(mean_angle.values[1:], np.pi / 4, atol=1e-10)

    def test_mean_angle_partial_alignment_matches_vector_average(
        self,
        partial_alignment_positions,
    ):
        """Mean angle matches vector average for partial alignment."""
        _, mean_angle = kinematics.compute_polarization(
            partial_alignment_positions,
            return_angle=True,
        )
        expected = np.arctan2(1, 2)
        assert np.allclose(mean_angle.values[1:], expected, atol=1e-10)

    def test_mean_angle_is_nan_when_net_vector_cancels(
        self,
        opposite_positions,
        perpendicular_positions,
    ):
        """Mean angle is NaN when heading vectors cancel out."""
        pol_opposite, angle_opposite = kinematics.compute_polarization(
            opposite_positions,
            return_angle=True,
        )
        pol_perp, angle_perp = kinematics.compute_polarization(
            perpendicular_positions,
            return_angle=True,
        )
        assert np.allclose(pol_opposite.values[1:], 0.0, atol=1e-10)
        assert np.allclose(pol_perp.values[1:], 0.0, atol=1e-10)
        assert np.all(np.isnan(angle_opposite.values[1:]))
        assert np.all(np.isnan(angle_perp.values[1:]))

    def test_mean_angle_rotates_with_global_rotation(
        self,
        partial_alignment_positions,
    ):
        """Mean angle shifts by the same amount under global rotation."""
        _, angle_original = kinematics.compute_polarization(
            partial_alignment_positions,
            return_angle=True,
        )

        x = partial_alignment_positions.sel(space="x")
        y = partial_alignment_positions.sel(space="y")

        rotated = partial_alignment_positions.copy()
        rotated.loc[{"space": "x"}] = -y
        rotated.loc[{"space": "y"}] = x

        _, angle_rotated = kinematics.compute_polarization(
            rotated,
            return_angle=True,
        )

        expected = angle_original.values[1:] + (np.pi / 2)
        expected = (expected + np.pi) % (2 * np.pi) - np.pi

        np.testing.assert_allclose(
            angle_rotated.values[1:],
            expected,
            atol=1e-10,
        )

    def test_mean_angle_wraparound_near_pi_is_handled_correctly(self):
        """Headings near +pi and -pi should average leftward, not to zero."""
        # Two individuals moving left with tiny y-offsets in opposite dirs.
        # This creates headings very close to +pi and -pi.
        data = np.array(
            [
                [[0.0, 0.0], [0.0, 0.0]],
                [[-1.0, -1.0], [1e-6, -1e-6]],
                [[-2.0, -2.0], [2e-6, -2e-6]],
            ],
            dtype=float,
        )

        polarization, mean_angle = kinematics.compute_polarization(
            _make_position_dataarray(data),
            return_angle=True,
        )

        assert np.allclose(polarization.values[1:], 1.0, atol=1e-10)
        assert np.allclose(
            np.abs(mean_angle.values[1:]),
            np.pi,
            atol=1e-6,
        )

    def test_in_degrees_true_returns_degrees(self):
        """in_degrees=True returns angle in degrees."""
        # Two individuals moving in +y direction
        data = np.array(
            [
                [[0, 0], [0, 0]],
                [[0, 0], [1, 1]],
                [[0, 0], [2, 2]],
            ],
            dtype=float,
        )
        _, mean_angle_rad = kinematics.compute_polarization(
            _make_position_dataarray(data),
            return_angle=True,
            in_degrees=False,
        )
        _, mean_angle_deg = kinematics.compute_polarization(
            _make_position_dataarray(data),
            return_angle=True,
            in_degrees=True,
        )
        # +y direction = 90 degrees = pi/2 radians
        assert np.allclose(mean_angle_rad.values[1:], np.pi / 2, atol=1e-10)
        assert np.allclose(mean_angle_deg.values[1:], 90.0, atol=1e-10)
