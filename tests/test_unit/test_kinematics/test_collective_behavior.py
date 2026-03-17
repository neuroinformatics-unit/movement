"""Unit tests for collective behavior metrics."""

from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
import xarray as xr

from movement.kinematics.collective_behavior import (
    compute_approach_tangent_velocity,
    compute_egocentric_angle,
    compute_group_spread,
    compute_leadership,
    compute_milling,
    compute_polarization,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_position(pos_array, space=("x", "y"), individuals=None, time=None):
    """Build a (time, space, individuals) DataArray from a numpy array.

    Parameters
    ----------
    pos_array
        Shape ``(n_time, n_space, n_individuals)``.
    """
    n_time, n_space, n_ind = pos_array.shape
    if individuals is None:
        individuals = [f"ind{i}" for i in range(n_ind)]
    if time is None:
        time = np.arange(n_time, dtype=float)
    return xr.DataArray(
        pos_array,
        dims=["time", "space", "individuals"],
        coords={
            "time": time,
            "space": list(space)[:n_space],
            "individuals": individuals,
        },
    )


def _make_position_with_keypoints(
    pos_array, keypoints=("kp0", "kp1"), space=("x", "y"), individuals=None
):
    """Build a (time, space, keypoints, individuals) DataArray."""
    n_time, n_space, n_kp, n_ind = pos_array.shape
    if individuals is None:
        individuals = [f"ind{i}" for i in range(n_ind)]
    return xr.DataArray(
        pos_array,
        dims=["time", "space", "keypoints", "individuals"],
        coords={
            "time": np.arange(n_time, dtype=float),
            "space": list(space)[:n_space],
            "keypoints": list(keypoints)[:n_kp],
            "individuals": individuals,
        },
    )


@pytest.fixture
def aligned_position():
    """Three individuals all moving in the +x direction at unit speed.

    Expected polarization ≈ 1.
    Expected milling ≈ 0 (no rotation).
    """
    n_time, n_ind = 20, 3
    t = np.arange(n_time, dtype=float)
    # x increases with time, y stays at 0 / 1 / 2 (different y offsets)
    pos = np.zeros((n_time, 2, n_ind))
    for i in range(n_ind):
        pos[:, 0, i] = t  # x = t
        pos[:, 1, i] = float(i)  # y = constant (no y motion)
    return _make_position(pos)


@pytest.fixture
def circular_position():
    """Four individuals arranged on a circle, each moving tangentially
    (counterclockwise). Expected milling = 1, polarization ~ 0.
    """
    n_time = 50
    n_ind = 4
    omega = 2 * np.pi / n_time  # one full revolution
    t = np.arange(n_time, dtype=float)
    pos = np.zeros((n_time, 2, n_ind))
    for i in range(n_ind):
        theta0 = i * np.pi / 2  # evenly spaced start angles
        pos[:, 0, i] = np.cos(omega * t + theta0)  # x
        pos[:, 1, i] = np.sin(omega * t + theta0)  # y
    return _make_position(pos)


@pytest.fixture
def two_individual_position():
    """Two individuals at constant positions (no motion)."""
    n_time = 10
    pos = np.zeros((n_time, 2, 2))
    pos[:, 0, 0] = 0.0  # ind0 at (0, 0)
    pos[:, 0, 1] = 5.0  # ind1 at (5, 0)
    return _make_position(pos)


# ---------------------------------------------------------------------------
# Tests: compute_polarization
# ---------------------------------------------------------------------------


class TestComputePolarization:
    """Tests for compute_polarization."""

    def test_aligned_group_gives_high_polarization(self, aligned_position):
        """All individuals moving in +x → polarization ≈ 1."""
        pol = compute_polarization(aligned_position)
        assert pol.name == "polarization"
        assert pol.dims == ("time",)
        # Velocity-based: differentiate gives (1, 0) for each individual
        # (at interior time points). Exclude boundaries (central differences
        # at endpoints may differ slightly).
        np.testing.assert_allclose(
            pol.isel(time=slice(1, -1)).values,
            np.ones(len(pol) - 2),
            atol=1e-6,
        )

    def test_circular_group_gives_low_polarization(self, circular_position):
        """Individuals on a circle → headings cancel → polarization ≈ 0."""
        pol = compute_polarization(circular_position)
        # With 4 individuals symmetrically placed, sum of unit vectors is ~0
        np.testing.assert_allclose(
            pol.isel(time=slice(1, -1)).values,
            np.zeros(len(pol) - 2),
            atol=0.15,  # allow some tolerance from discretisation
        )

    def test_output_shape_and_range(self, aligned_position):
        """Polarization output is in [0, 1] with correct shape."""
        pol = compute_polarization(aligned_position)
        assert pol.dims == ("time",)
        assert pol.sizes["time"] == aligned_position.sizes["time"]
        valid = pol.values[~np.isnan(pol.values)]
        assert np.all(valid >= 0.0)
        assert np.all(valid <= 1.0 + 1e-9)

    def test_keypoint_based_heading(self):
        """Keypoint-based heading: tail→nose in +x direction."""
        n_time, n_ind = 10, 2
        # Each individual's "tail" at (0, y), "nose" at (1, y): heading = +x
        pos = np.zeros((n_time, 2, 2, n_ind))  # (time, space, keypoints, ind)
        for i in range(n_ind):
            pos[:, 0, 0, i] = 0.0  # tail x
            pos[:, 0, 1, i] = 1.0  # nose x
            pos[:, 1, :, i] = float(i)  # y offsets, same for both keypoints
        da = _make_position_with_keypoints(pos, keypoints=("tail", "nose"))
        pol = compute_polarization(da, heading_keypoints=("tail", "nose"))
        np.testing.assert_allclose(pol.values, np.ones(n_time), atol=1e-6)

    def test_single_individual_returns_one(self):
        """A single valid individual always gives polarization = 1."""
        n_time = 10
        pos = np.zeros((n_time, 2, 1))
        pos[:, 0, 0] = np.arange(n_time, dtype=float)
        da = _make_position(pos)
        pol = compute_polarization(da)
        # Interior time points: velocity = (1, 0), unit vector, norm = 1
        np.testing.assert_allclose(
            pol.isel(time=slice(1, -1)).values,
            np.ones(n_time - 2),
            atol=1e-6,
        )

    def test_all_nan_individuals_gives_nan(self):
        """All NaN positions → NaN polarization."""
        n_time, n_ind = 5, 2
        pos = np.full((n_time, 2, n_ind), np.nan)
        da = _make_position(pos)
        pol = compute_polarization(da)
        assert np.all(np.isnan(pol.values))

    def test_invalid_input_missing_dim(self):
        """Missing 'individuals' dimension raises ValueError."""
        da = xr.DataArray(
            np.ones((5, 2)), dims=["time", "space"],
            coords={"time": np.arange(5.0), "space": ["x", "y"]},
        )
        with pytest.raises(ValueError, match="individuals"):
            compute_polarization(da)

    def test_invalid_keypoint_name(self):
        """Unknown keypoint name raises ValueError."""
        n_time, n_ind = 5, 2
        pos = np.zeros((n_time, 2, 2, n_ind))
        da = _make_position_with_keypoints(pos, keypoints=("tail", "nose"))
        with pytest.raises(ValueError):
            compute_polarization(da, heading_keypoints=("tail", "snout"))


# ---------------------------------------------------------------------------
# Tests: compute_milling
# ---------------------------------------------------------------------------


class TestComputeMilling:
    """Tests for compute_milling."""

    def test_circular_motion_gives_high_milling(self, circular_position):
        """Uniform circular motion → milling ≈ 1."""
        mill = compute_milling(circular_position)
        assert mill.name == "milling"
        assert mill.dims == ("time",)
        np.testing.assert_allclose(
            mill.isel(time=slice(1, -1)).values,
            np.ones(len(mill) - 2),
            atol=0.05,
        )

    def test_aligned_motion_gives_low_milling(self, aligned_position):
        """Aligned linear motion → angular momenta cancel → milling ≈ 0.

        For individuals at different y-positions moving in +x, the cross
        products have opposite signs and roughly cancel for symmetric pairs.
        """
        mill = compute_milling(aligned_position)
        valid = mill.values[~np.isnan(mill.values)]
        assert valid.shape[0] > 0
        assert np.all(valid >= 0.0)
        assert np.all(valid <= 1.0 + 1e-9)

    def test_output_range(self, circular_position):
        """Milling values are always in [0, 1]."""
        mill = compute_milling(circular_position)
        valid = mill.values[~np.isnan(mill.values)]
        assert np.all(valid >= 0.0)
        assert np.all(valid <= 1.0 + 1e-9)

    def test_single_individual_is_nan_or_zero(self):
        """Single stationary individual → zero cross product → NaN."""
        n_time = 10
        pos = np.zeros((n_time, 2, 1))
        da = _make_position(pos)
        mill = compute_milling(da)
        assert np.all(np.isnan(mill.values))

    def test_3d_space_raises(self):
        """3D position (x, y, z) raises ValueError for milling."""
        n_time, n_ind = 5, 2
        pos_3d = np.ones((n_time, 3, n_ind))
        da = xr.DataArray(
            pos_3d,
            dims=["time", "space", "individuals"],
            coords={
                "time": np.arange(n_time, dtype=float),
                "space": ["x", "y", "z"],
                "individuals": [f"ind{i}" for i in range(n_ind)],
            },
        )
        with pytest.raises(ValueError):
            compute_milling(da)

    def test_keypoint_based_heading(self):
        """Milling with keypoint-based heading runs without error."""
        n_time, n_kp, n_ind = 15, 2, 3
        pos = np.zeros((n_time, 2, n_kp, n_ind))
        # Create circular motion
        t = np.arange(n_time, dtype=float)
        omega = 2 * np.pi / n_time
        for i in range(n_ind):
            theta0 = i * 2 * np.pi / n_ind
            pos[:, 0, :, i] = np.cos(omega * t + theta0)[:, None]
            pos[:, 1, :, i] = np.sin(omega * t + theta0)[:, None]
        # Offset the two keypoints slightly to create a heading
        pos[:, 0, 1, :] += 0.1  # nose slightly ahead in x

        da = _make_position_with_keypoints(pos)
        mill = compute_milling(da, heading_keypoints=("kp0", "kp1"))
        assert mill.dims == ("time",)


# ---------------------------------------------------------------------------
# Tests: compute_group_spread
# ---------------------------------------------------------------------------


class TestComputeGroupSpread:
    """Tests for compute_group_spread."""

    def test_known_spread_value(self):
        """Three individuals at known positions → exact spread value."""
        # Place 3 individuals at (0,0), (2,0), (4,0)
        # centroid = (2, 0)
        # distances: 2, 0, 2 → RMS = sqrt((4+0+4)/3) = sqrt(8/3)
        n_time, n_ind = 5, 3
        pos = np.zeros((n_time, 2, n_ind))
        pos[:, 0, 0] = 0.0
        pos[:, 0, 1] = 2.0
        pos[:, 0, 2] = 4.0
        da = _make_position(pos)
        spread = compute_group_spread(da)
        expected = np.sqrt(8.0 / 3.0)
        np.testing.assert_allclose(spread.values, expected, atol=1e-6)

    def test_output_name_and_dims(self):
        """Output has correct name and dimensions."""
        n_time, n_ind = 10, 3
        pos = np.random.default_rng(42).random((n_time, 2, n_ind))
        da = _make_position(pos)
        spread = compute_group_spread(da)
        assert spread.name == "group_spread"
        assert spread.dims == ("time",)

    def test_collocated_individuals_gives_zero(self):
        """All individuals at the same location → spread = 0."""
        n_time, n_ind = 5, 4
        pos = np.zeros((n_time, 2, n_ind))
        da = _make_position(pos)
        spread = compute_group_spread(da)
        np.testing.assert_allclose(spread.values, 0.0, atol=1e-9)

    def test_single_individual_gives_nan(self):
        """One individual → fewer than 2 → NaN."""
        n_time = 5
        pos = np.ones((n_time, 2, 1))
        da = _make_position(pos)
        spread = compute_group_spread(da)
        assert np.all(np.isnan(spread.values))

    def test_nan_positions_excluded(self):
        """NaN positions are excluded from spread calculation."""
        n_time, n_ind = 5, 3
        pos = np.zeros((n_time, 2, n_ind))
        pos[:, 0, 0] = 0.0
        pos[:, 0, 1] = 2.0
        pos[:, :, 2] = np.nan  # third individual missing
        da = _make_position(pos)
        spread = compute_group_spread(da)
        # With only 2 valid individuals at (0,0) and (2,0):
        # centroid = (1, 0), distances both 1 → spread = 1
        np.testing.assert_allclose(spread.values, 1.0, atol=1e-6)

    def test_unsupported_method_raises(self):
        """Unsupported method name raises ValueError."""
        n_time, n_ind = 5, 2
        pos = np.zeros((n_time, 2, n_ind))
        da = _make_position(pos)
        with pytest.raises(ValueError, match="radius_of_gyration"):
            compute_group_spread(da, method="convex_hull")

    def test_spread_increases_with_separation(self):
        """Larger separation → larger spread."""
        n_time, n_ind = 5, 2
        pos_close = np.zeros((n_time, 2, n_ind))
        pos_close[:, 0, 1] = 1.0
        pos_far = np.zeros((n_time, 2, n_ind))
        pos_far[:, 0, 1] = 10.0
        spread_close = compute_group_spread(_make_position(pos_close))
        spread_far = compute_group_spread(_make_position(pos_far))
        assert float(spread_far.mean()) > float(spread_close.mean())


# ---------------------------------------------------------------------------
# Tests: compute_leadership
# ---------------------------------------------------------------------------


class TestComputeLeadership:
    """Tests for compute_leadership."""

    @pytest.fixture
    def leader_follower_position(self):
        """Individual 'A' leads individual 'B' by 5 frames.

        Positions are cubic functions of time so that the velocity signal
        is a clear quadratic, making cross-correlation unambiguous.
        """
        lag = 5
        n_time = 80
        t = np.arange(n_time, dtype=float)
        pos = np.zeros((n_time, 2, 2))
        pos[:, 0, 0] = t**3 / 3.0  # A: pos_x = t^3/3, vel_x ≈ t^2
        pos[:, 0, 1] = (t - lag) ** 3 / 3.0  # B: same but delayed
        return _make_position(pos, individuals=["A", "B"]), lag

    def test_output_structure(self, leader_follower_position):
        """Output has correct dims, coords, and name."""
        pos, _ = leader_follower_position
        result = compute_leadership(pos, max_lag=10)
        assert result.name == "leadership"
        assert set(result.dims) == {"individuals", "individuals_other", "metric"}
        assert list(result.coords["metric"].values) == ["correlation", "lag"]
        np.testing.assert_array_equal(
            result.coords["individuals"].values, ["A", "B"]
        )

    def test_diagonal_is_nan(self, leader_follower_position):
        """Self-pairs (diagonal) must be NaN."""
        pos, _ = leader_follower_position
        result = compute_leadership(pos, max_lag=10)
        for ind in ["A", "B"]:
            assert np.isnan(
                float(result.sel(individuals=ind, individuals_other=ind,
                                 metric="correlation"))
            )

    def test_leader_has_positive_lag(self, leader_follower_position):
        """A leads B → lag[A,B] > 0; B follows A → lag[B,A] < 0."""
        pos, true_lag = leader_follower_position
        result = compute_leadership(pos, max_lag=true_lag + 3)
        lag_ab = float(result.sel(
            individuals="A", individuals_other="B", metric="lag"
        ))
        lag_ba = float(result.sel(
            individuals="B", individuals_other="A", metric="lag"
        ))
        assert lag_ab > 0, f"Expected positive lag for A→B, got {lag_ab}"
        assert lag_ba < 0, f"Expected negative lag for B→A, got {lag_ba}"

    def test_lag_close_to_true_lag(self, leader_follower_position):
        """Detected lag should be within 1 frame of the true lag."""
        pos, true_lag = leader_follower_position
        result = compute_leadership(pos, max_lag=true_lag + 3)
        lag_ab = float(result.sel(
            individuals="A", individuals_other="B", metric="lag"
        ))
        assert abs(lag_ab - true_lag) <= 1, (
            f"Expected lag ≈ {true_lag}, got {lag_ab}"
        )

    def test_invalid_max_lag_raises(self, leader_follower_position):
        """Non-positive max_lag raises ValueError."""
        pos, _ = leader_follower_position
        with pytest.raises(ValueError, match="max_lag"):
            compute_leadership(pos, max_lag=0)
        with pytest.raises((ValueError, TypeError)):
            compute_leadership(pos, max_lag=-5)

    def test_high_correlation_for_true_lag(self, leader_follower_position):
        """Correlation at optimal lag should be close to 1."""
        pos, true_lag = leader_follower_position
        result = compute_leadership(pos, max_lag=true_lag + 3)
        corr_ab = float(result.sel(
            individuals="A", individuals_other="B", metric="correlation"
        ))
        assert corr_ab > 0.95, f"Expected high correlation, got {corr_ab}"

    def test_with_keypoints(self):
        """Leadership computation works when position has keypoints dim."""
        n_time, n_kp, n_ind = 50, 2, 2
        pos = np.zeros((n_time, 2, n_kp, n_ind))
        t = np.arange(n_time, dtype=float)
        pos[:, 0, :, 0] = t[:, None] ** 2
        pos[:, 0, :, 1] = (t - 3)[:, None] ** 2
        da = _make_position_with_keypoints(pos)
        result = compute_leadership(da, max_lag=6)
        assert result.dims == ("individuals", "individuals_other", "metric")


# ---------------------------------------------------------------------------
# Tests: compute_egocentric_angle
# ---------------------------------------------------------------------------


class TestComputeEgocentricAngle:
    """Tests for compute_egocentric_angle."""

    def test_other_directly_ahead_gives_zero(self):
        """Other individual directly in front of focal → angle ≈ 0."""
        n_time = 5
        # ind0 moves from x=0 to x=4; ind1 stays at x=100 (always ahead)
        pos = np.zeros((n_time, 2, 2))
        pos[:, 0, 0] = np.arange(n_time, dtype=float)  # ind0 moves +x
        pos[:, 0, 1] = 100.0  # ind1 far ahead in +x, never overtaken

        da = _make_position(pos)
        angle = compute_egocentric_angle(da)
        # At interior time steps, heading of ind0 is (1, 0)
        # vec_to_other = (100 - t, 0) which is always positive → angle = 0
        interior = angle.isel(time=slice(1, -1))
        ang_0_to_1 = interior.sel(individuals="ind0", individuals_other="ind1")
        np.testing.assert_allclose(
            ang_0_to_1.values, 0.0, atol=1e-5
        )

    def test_other_directly_behind_gives_pi(self):
        """Other individual directly behind focal → |angle| ≈ π."""
        n_time = 5
        pos = np.zeros((n_time, 2, 2))
        pos[:, 0, 0] = np.arange(n_time, dtype=float)  # ind0 moves +x
        pos[:, 0, 1] = -2.0  # ind1 is behind ind0 at x=-2

        da = _make_position(pos)
        angle = compute_egocentric_angle(da)
        interior = angle.isel(time=slice(1, -1))
        ang_0_to_1 = interior.sel(individuals="ind0", individuals_other="ind1")
        np.testing.assert_allclose(
            np.abs(ang_0_to_1.values), np.pi, atol=1e-5
        )

    def test_other_to_the_left_gives_positive_angle(self):
        """Other individual to the left of heading → positive angle."""
        n_time = 5
        pos = np.zeros((n_time, 2, 2))
        pos[:, 0, 0] = np.arange(n_time, dtype=float)  # ind0 moves +x
        pos[:, 1, 1] = 2.0  # ind1 is at y=2 (left in image coords y-down)

        da = _make_position(pos)
        angle = compute_egocentric_angle(da)
        interior = angle.isel(time=slice(1, -1))
        ang_0_to_1 = interior.sel(individuals="ind0", individuals_other="ind1")
        # vec_to_other = (0-t, 2) for each t; cross with (1,0) = 2 > 0 → positive angle
        assert np.all(ang_0_to_1.values > 0)

    def test_self_pairs_are_nan(self):
        """Diagonal (self-pairs) should be NaN."""
        n_time, n_ind = 5, 3
        pos = np.zeros((n_time, 2, n_ind))
        for i in range(n_ind):
            pos[:, 0, i] = np.arange(n_time, dtype=float) + i
        da = _make_position(pos)
        angle = compute_egocentric_angle(da)
        for ind in da.coords["individuals"].values:
            assert np.all(
                np.isnan(
                    angle.sel(individuals=ind, individuals_other=ind).values
                )
            ), f"Self-pair for {ind} is not NaN"

    def test_output_shape(self):
        """Output has correct dims and sizes."""
        n_time, n_ind = 10, 3
        pos = np.zeros((n_time, 2, n_ind))
        for i in range(n_ind):
            pos[:, 0, i] = np.arange(n_time, dtype=float) * (i + 1)
        da = _make_position(pos)
        angle = compute_egocentric_angle(da)
        assert angle.dims == ("time", "individuals", "individuals_other")
        assert angle.sizes == {
            "time": n_time,
            "individuals": n_ind,
            "individuals_other": n_ind,
        }

    def test_in_degrees_conversion(self):
        """in_degrees=True returns degrees."""
        n_time, n_ind = 5, 2
        pos = np.zeros((n_time, 2, n_ind))
        pos[:, 0, 0] = np.arange(n_time, dtype=float)
        pos[:, 0, 1] = np.arange(n_time, dtype=float) + 2.0
        da = _make_position(pos)
        angle_rad = compute_egocentric_angle(da, in_degrees=False)
        angle_deg = compute_egocentric_angle(da, in_degrees=True)
        np.testing.assert_allclose(
            np.deg2rad(angle_deg.values),
            angle_rad.values,
            atol=1e-9,
            equal_nan=True,
        )

    def test_output_range(self):
        """All non-NaN angles are in (-pi, pi]."""
        n_time, n_ind = 15, 4
        rng = np.random.default_rng(0)
        pos = rng.random((n_time, 2, n_ind))
        # add linear motion so heading is defined
        for i in range(n_ind):
            pos[:, 0, i] += np.arange(n_time, dtype=float) * 0.1
        da = _make_position(pos)
        angle = compute_egocentric_angle(da)
        valid = angle.values[~np.isnan(angle.values)]
        assert np.all(valid > -np.pi - 1e-9)
        assert np.all(valid <= np.pi + 1e-9)

    def test_keypoint_based_heading(self):
        """Runs correctly with keypoint-based heading."""
        n_time, n_kp, n_ind = 10, 2, 3
        pos = np.zeros((n_time, 2, n_kp, n_ind))
        for i in range(n_ind):
            pos[:, 0, :, i] = np.arange(n_time, dtype=float)[:, None] * (i + 1)
        pos[:, 0, 1, :] += 0.5  # nose slightly ahead
        da = _make_position_with_keypoints(pos)
        angle = compute_egocentric_angle(da, heading_keypoints=("kp0", "kp1"))
        assert angle.dims == ("time", "individuals", "individuals_other")

    def test_3d_space_raises(self):
        """3D position raises ValueError for egocentric_angle."""
        n_time, n_ind = 5, 2
        pos = np.ones((n_time, 3, n_ind))
        da = xr.DataArray(
            pos,
            dims=["time", "space", "individuals"],
            coords={
                "time": np.arange(n_time, dtype=float),
                "space": ["x", "y", "z"],
                "individuals": ["ind0", "ind1"],
            },
        )
        with pytest.raises(ValueError):
            compute_egocentric_angle(da)


# ---------------------------------------------------------------------------
# Tests: compute_approach_tangent_velocity
# ---------------------------------------------------------------------------


class TestComputeApproachTangentVelocity:
    """Tests for compute_approach_tangent_velocity."""

    def test_head_on_approach_radial_negative(self):
        """Two individuals approaching head-on → radial velocity < 0.

        Radial is positive when they move apart; negative when approaching.
        """
        n_time = 10
        t = np.arange(n_time, dtype=float)
        pos = np.zeros((n_time, 2, 2))
        pos[:, 0, 0] = t           # ind0 moves right (+x)
        pos[:, 0, 1] = 20.0 - t   # ind1 moves left (-x)
        da = _make_position(pos)
        atv = compute_approach_tangent_velocity(da)
        radial = atv.sel(
            individuals="ind0", individuals_other="ind1", component="radial"
        )
        # At interior points, relative velocity = vel1 - vel0 = -1 - 1 = -2
        # unit_radial points from ind0 to ind1 (in +x direction roughly)
        # dot(-2, +1) = -2 < 0  (not apart but approaching... wait)
        # Actually: radial = dot(rel_vel, unit_radial) with unit_radial from i to j
        # unit_radial from ind0 to ind1 is in +x (initially)
        # rel_vel = vel1 - vel0 = (-1) - (1) = (-2, 0)
        # dot((-2, 0), (1, 0)) = -2 → negative, distance decreasing ✓
        np.testing.assert_array_less(
            radial.isel(time=slice(1, -1)).values,
            np.zeros(n_time - 2) + 1e-9,
        )

    def test_parallel_motion_tangential_near_zero(self):
        """Two individuals moving in parallel → tangential ≈ 0."""
        n_time = 10
        t = np.arange(n_time, dtype=float)
        pos = np.zeros((n_time, 2, 2))
        pos[:, 0, 0] = t      # ind0 moves +x
        pos[:, 0, 1] = t      # ind1 moves +x (parallel, same velocity)
        pos[:, 1, 1] = 2.0    # ind1 offset in y
        da = _make_position(pos)
        atv = compute_approach_tangent_velocity(da)
        tangential = atv.sel(
            individuals="ind0", individuals_other="ind1", component="tangential"
        )
        # relative velocity = (0, 0) → tangential = 0
        np.testing.assert_allclose(
            tangential.isel(time=slice(1, -1)).values,
            0.0,
            atol=1e-6,
        )

    def test_output_structure(self):
        """Output has correct dims, coords, and name."""
        n_time, n_ind = 10, 3
        rng = np.random.default_rng(1)
        pos = rng.random((n_time, 2, n_ind))
        for i in range(n_ind):
            pos[:, 0, i] += np.arange(n_time, dtype=float) * 0.1
        da = _make_position(pos)
        atv = compute_approach_tangent_velocity(da)
        assert atv.name == "approach_tangent_velocity"
        assert atv.dims == (
            "time", "individuals", "individuals_other", "component"
        )
        assert list(atv.coords["component"].values) == ["radial", "tangential"]

    def test_self_pairs_radial_is_nan(self):
        """Self-pair radial component should be NaN (zero relative distance)."""
        n_time, n_ind = 5, 3
        pos = np.zeros((n_time, 2, n_ind))
        for i in range(n_ind):
            pos[:, :, i] = np.arange(n_time, dtype=float)[:, None]
        da = _make_position(pos)
        atv = compute_approach_tangent_velocity(da)
        for ind in da.coords["individuals"].values:
            radial = atv.sel(
                individuals=ind, individuals_other=ind, component="radial"
            )
            assert np.all(np.isnan(radial.values)), (
                f"Self-pair radial for {ind} is not NaN"
            )

    def test_tangential_nonnegative(self):
        """Tangential velocity (magnitude) is always non-negative."""
        n_time, n_ind = 10, 3
        rng = np.random.default_rng(2)
        pos = rng.random((n_time, 2, n_ind))
        for i in range(n_ind):
            pos[:, 0, i] += np.arange(n_time, dtype=float) * 0.2 * (i + 1)
        da = _make_position(pos)
        atv = compute_approach_tangent_velocity(da)
        tangential = atv.sel(component="tangential")
        valid = tangential.values[~np.isnan(tangential.values)]
        assert np.all(valid >= -1e-10)

    def test_with_keypoints(self):
        """Works when position has keypoints dimension."""
        n_time, n_kp, n_ind = 10, 2, 2
        pos = np.zeros((n_time, 2, n_kp, n_ind))
        t = np.arange(n_time, dtype=float)
        pos[:, 0, :, 0] = t[:, None]
        pos[:, 0, :, 1] = (10.0 - t)[:, None]
        da = _make_position_with_keypoints(pos)
        atv = compute_approach_tangent_velocity(da)
        assert atv.dims == (
            "time", "individuals", "individuals_other", "component"
        )

    def test_moving_apart_radial_positive(self):
        """Two individuals moving apart → radial velocity > 0."""
        n_time = 10
        t = np.arange(n_time, dtype=float)
        pos = np.zeros((n_time, 2, 2))
        pos[:, 0, 0] = -t      # ind0 moves -x
        pos[:, 0, 1] = 5.0 + t  # ind1 moves +x, start further away
        da = _make_position(pos)
        atv = compute_approach_tangent_velocity(da)
        radial = atv.sel(
            individuals="ind0", individuals_other="ind1", component="radial"
        )
        # rel_vel = vel1 - vel0 = (1) - (-1) = (2, 0)
        # unit_radial from ind0 to ind1 is in +x direction
        # dot((2, 0), (1, 0)) = 2 > 0 → positive, distance increasing
        np.testing.assert_array_less(
            np.zeros(n_time - 2) - 1e-9,
            radial.isel(time=slice(1, -1)).values,
        )
