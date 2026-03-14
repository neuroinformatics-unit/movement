"""Unit tests for ``compute_inactivity_bouts``."""

import numpy as np
import pytest
import xarray as xr

from movement.kinematics import compute_inactivity_bouts


def _make_position(
    values: np.ndarray,
    time_unit: float = 1.0,
    n_individuals: int = 1,
) -> xr.DataArray:
    """Build a minimal 2-D (x, y) position DataArray.

    Parameters
    ----------
    values
        Array of shape ``(n_time, 2)`` holding (x, y) positions for a
        single individual.  For multiple individuals the same trajectory
        is repeated.
    time_unit
        Spacing between consecutive time points (default 1 frame = 1 s).
    n_individuals
        Number of individuals (each gets the same trajectory).

    Returns
    -------
    xarray.DataArray
        Position array with dimensions ``(time, space, individuals)``.

    """
    n_time = len(values)
    data = np.stack([values] * n_individuals, axis=-1)  # (time, 2, indiv)
    return xr.DataArray(
        data,
        dims=["time", "space", "individuals"],
        coords={
            "time": np.arange(n_time, dtype=float) * time_unit,
            "space": ["x", "y"],
            "individuals": [f"id_{i}" for i in range(n_individuals)],
        },
    )


# ------------------------------------------------------------------ helpers
@pytest.fixture()
def stationary_position():
    """10-frame position that never moves (speed = 0 everywhere)."""
    return _make_position(np.zeros((10, 2)))


@pytest.fixture()
def moving_position():
    """10-frame position with constant velocity (speed >> 0 everywhere)."""
    t = np.arange(10)
    return _make_position(np.column_stack([t, t]))  # moves at sqrt(2)/frame


@pytest.fixture()
def mixed_position():
    """Position: still for frames 0-2, moving frames 3-6, still frames 7-9.

    Speed == 0 at frames 0-2 and 7-9; non-zero at frames 3-6.
    Two inactivity bouts are expected.
    """
    pos = np.zeros((10, 2))
    # inject movement in the middle block
    pos[3:7, 0] = np.arange(1, 5)
    pos[3:7, 1] = np.arange(1, 5)
    return _make_position(pos)


# ============================================================== basic tests
class TestComputeInactivityBoutsBasic:
    """Basic correctness checks for ``compute_inactivity_bouts``."""

    def test_stationary_single_bout(self, stationary_position):
        """A completely stationary individual produces exactly one bout."""
        result = compute_inactivity_bouts(
            stationary_position, speed_threshold=1.0
        )
        assert result.name == "inactivity_bout_id"
        assert "time" in result.dims
        # All frames belong to a single bout
        bout_ids = result.isel(individuals=0).values
        assert set(bout_ids) == {1}

    def test_moving_no_bouts(self, moving_position):
        """No inactivity bouts for a fast-moving individual (all zeros)."""
        result = compute_inactivity_bouts(moving_position, speed_threshold=0.5)
        assert (result == 0).all()

    def test_two_bouts_in_mixed_trajectory(self, mixed_position):
        """Still–moving–still trajectory yields exactly two bouts."""
        result = compute_inactivity_bouts(mixed_position, speed_threshold=0.5)
        bout_ids = result.isel(individuals=0).values
        unique_ids = sorted(set(bout_ids[bout_ids > 0]))
        assert unique_ids == [1, 2], (
            f"Expected 2 bouts, got unique IDs {unique_ids}"
        )

    def test_bout_ids_ordered_by_onset(self, mixed_position):
        """Bout IDs increase monotonically across time."""
        result = compute_inactivity_bouts(mixed_position, speed_threshold=0.5)
        bout_ids = result.isel(individuals=0).values
        # The first non-zero id encountered should be 1, the second 2, ...
        seen = []
        for v in bout_ids:
            if v > 0 and v not in seen:
                seen.append(v)
        assert seen == sorted(seen)

    def test_output_dimensions_match_speed(self, moving_position):
        """Output shape equals speed shape (space dim removed)."""
        result = compute_inactivity_bouts(moving_position, speed_threshold=0.5)
        assert "space" not in result.dims
        assert "time" in result.dims
        assert "individuals" in result.dims

    def test_output_dtype_is_integer(self, stationary_position):
        """Output values must be integers."""
        result = compute_inactivity_bouts(
            stationary_position, speed_threshold=1.0
        )
        assert np.issubdtype(result.dtype, np.integer)

    def test_zero_threshold_no_bouts(self, moving_position):
        """With threshold=0, no frame can be inactive (speed >= 0)."""
        result = compute_inactivity_bouts(moving_position, speed_threshold=0.0)
        assert (result == 0).all()


# =========================================== min_bout_duration tests
class TestMinBoutDuration:
    """Tests for the ``min_bout_duration`` parameter."""

    def test_min_duration_suppresses_short_bout(self, mixed_position):
        """Bouts shorter than min_bout_duration are set to 0.

        The ``mixed_position`` fixture has bouts of duration ≥ 2 s each
        (3 and 3 frames at 1 frame/s).  A min_bout_duration slightly above
        the bout span should suppress both.
        """
        # Each stationary block spans 3 frames → duration = 2 s (last - first)
        result = compute_inactivity_bouts(
            mixed_position, speed_threshold=0.5, min_bout_duration=3.0
        )
        # All bouts should be suppressed
        assert (result == 0).all()

    def test_min_duration_keeps_long_bout(self, stationary_position):
        """A long bout is kept when min_bout_duration is below its duration."""
        # 10 stationary frames at 1 s/frame → duration = 9 s
        result = compute_inactivity_bouts(
            stationary_position, speed_threshold=1.0, min_bout_duration=5.0
        )
        assert (result > 0).any()

    def test_zero_min_duration_keeps_single_frame_bout(self):
        """Single-frame bout is retained for min_bout_duration=0.0.

        Create a symmetric "pause": uniform motion except at frame 5,
        where pos[4] == pos[6], forcing central-difference velocity=0.
        This produces exactly one isolated inactive frame (frame 5) with
        bout duration=0 s.  It must be kept when min_bout_duration=0.0.
        """
        pos = np.arange(10, dtype=float).reshape(-1, 1)
        pos = pos * np.array([[5.0, 5.0]])
        pos[6, :] = pos[4, :]  # zero central-diff velocity at frame 5
        position = _make_position(pos)
        result = compute_inactivity_bouts(
            position, speed_threshold=0.5, min_bout_duration=0.0
        )
        assert result.isel(time=5, individuals=0).item() == 1

    def test_positive_min_duration_drops_single_frame_bout(self):
        """Single-frame bout is removed for any positive min_bout_duration."""
        pos = np.arange(10, dtype=float).reshape(-1, 1)
        pos = pos * np.array([[5.0, 5.0]])
        pos[6, :] = pos[4, :]  # isolated zero-speed frame at index 5
        position = _make_position(pos)
        result = compute_inactivity_bouts(
            position, speed_threshold=0.5, min_bout_duration=0.5
        )
        assert result.isel(time=5, individuals=0).item() == 0

    def test_non_default_time_unit(self):
        """Duration filtering works correctly for non-integer time steps.

        With time_unit=0.1 s and positions[0..4]=[0,0] (stationary),
        frames 0-3 have zero central-difference velocity (frame 4 uses
        frame 5's fast movement so its speed is high).  The inactive
        bout spans frames 0-3, giving duration = time[3]-time[0] = 0.3 s.
        """
        pos = np.zeros((10, 2))
        # fast movement starting at frame 5
        pos[5:, 0] = np.arange(1, 6) * 50.0
        pos[5:, 1] = np.arange(1, 6) * 50.0
        position = _make_position(pos, time_unit=0.1)

        # min_bout_duration < 0.3 s → bout kept
        result_kept = compute_inactivity_bouts(
            position, speed_threshold=1.0, min_bout_duration=0.25
        )
        assert (result_kept > 0).any()

        # min_bout_duration > 0.3 s → bout dropped
        result_dropped = compute_inactivity_bouts(
            position, speed_threshold=1.0, min_bout_duration=0.35
        )
        assert (result_dropped == 0).all()


# ============================================================== NaN handling
class TestNaNHandling:
    """Tests for frames with NaN position data."""

    def test_nan_frames_treated_as_active(self):
        """Frames with NaN position are treated as active (not inactive)."""
        pos = np.zeros((10, 2))
        pos[3, :] = np.nan  # one NaN frame
        position = _make_position(pos)

        result = compute_inactivity_bouts(position, speed_threshold=1.0)
        # The NaN frame should NOT be labelled as part of an inactivity bout
        assert result.isel(time=3, individuals=0).item() == 0

    def test_all_nan_no_bouts(self):
        """Fully NaN track produces no inactivity bouts at all."""
        pos = np.full((10, 2), np.nan)
        position = _make_position(pos)
        result = compute_inactivity_bouts(position, speed_threshold=99.0)
        assert (result == 0).all()


# ========================================== multiple individuals
class TestMultipleIndividuals:
    """Tests with multiple individuals in one DataArray."""

    def test_independent_labelling_per_individual(self):
        """Bout IDs are assigned independently for each individual."""
        pos = np.zeros((10, 2))
        position = _make_position(pos, n_individuals=3)
        result = compute_inactivity_bouts(position, speed_threshold=1.0)
        # Every individual should have the same bout (id=1 throughout)
        for i in range(3):
            ids = result.isel(individuals=i).values
            assert set(ids) == {1}

    def test_mixed_activity_across_individuals(self):
        """Two individuals can have different bout patterns."""
        still = np.zeros((10, 2))
        moving = np.column_stack([np.arange(10) * 5.0, np.arange(10) * 5.0])
        data = np.stack([still, moving], axis=-1)
        position = xr.DataArray(
            data,
            dims=["time", "space", "individuals"],
            coords={
                "time": np.arange(10, dtype=float),
                "space": ["x", "y"],
                "individuals": ["still", "moving"],
            },
        )
        result = compute_inactivity_bouts(position, speed_threshold=1.0)
        # still individual has active bouts; moving individual has none
        assert (result.sel(individuals="still") > 0).any()
        assert (result.sel(individuals="moving") == 0).all()


# ============================================================ invalid inputs
class TestInvalidInputs:
    """Tests that invalid arguments raise appropriate errors."""

    def test_negative_speed_threshold_raises(self, stationary_position):
        """A negative speed_threshold must raise a ValueError."""
        with pytest.raises(ValueError, match="speed_threshold"):
            compute_inactivity_bouts(stationary_position, speed_threshold=-1.0)

    def test_negative_min_bout_duration_raises(self, stationary_position):
        """A negative min_bout_duration must raise a ValueError."""
        with pytest.raises(ValueError, match="min_bout_duration"):
            compute_inactivity_bouts(
                stationary_position,
                speed_threshold=1.0,
                min_bout_duration=-0.5,
            )

    def test_missing_time_dim_raises(self):
        """Data without a ``time`` dimension must raise a ValueError."""
        bad = xr.DataArray(np.ones((3, 2)), dims=["space", "individuals"])
        with pytest.raises(ValueError):
            compute_inactivity_bouts(bad, speed_threshold=1.0)

    def test_missing_space_dim_raises(self):
        """Data without a ``space`` dimension must raise a ValueError."""
        bad = xr.DataArray(np.ones((10, 2)), dims=["time", "individuals"])
        with pytest.raises(ValueError):
            compute_inactivity_bouts(bad, speed_threshold=1.0)
