import numpy as np
import pytest
import xarray as xr

from movement.roi import compute_entry_exits, compute_region_occupancy


@pytest.mark.parametrize(
    "region_fixtures, data, expected_output",
    [
        pytest.param(
            ["triangle", "unit_square", "unit_square_with_hole"],
            np.array([[0.15, 0.15], [0.85, 0.85], [0.5, 0.5], [1.5, 1.5]]),
            {
                "data": np.array(
                    [
                        [True, False, True, False],
                        [True, True, True, False],
                        [True, True, False, False],
                    ]
                ),
                "coords": ["triangle", "Unit square", "Unit square with hole"],
            },
            id="triangle, unit_square, unit_square_with_hole",
        ),
        pytest.param(
            ["triangle", "triangle", "triangle"],
            np.array([[0.15, 0.15], [0.85, 0.85], [0.5, 0.5], [1.5, 1.5]]),
            {
                "data": np.array([[True, False, True, False]] * 3),
                "coords": ["triangle_0", "triangle_1", "triangle_2"],
            },
            id="3 superimposed triangles with same name",
        ),
        pytest.param(
            ["triangle", "triangle_different_name"],
            np.array([[0.5, 0.5]]),
            {
                "data": np.array([True] * 2),
                "coords": ["triangle", "pizza_slice"],
            },
            id="2 superimposed triangles with different names",
        ),
        pytest.param(
            ["triangle", "triangle_moved_01", "triangle_moved_100"],
            np.array([[0.5, 0.5]]),
            {
                "data": np.array([[True], [True], [False]]),
                "coords": ["triangle_0", "triangle_1", "triangle_2"],
            },
            id="3 different triangles with same name",
        ),
        pytest.param(
            ["triangle_different_name"],
            np.array([[0.5, 0.5]]),
            {
                "data": np.array([[True]]),
                "coords": ["pizza_slice"],
            },
            id="1 pizza slice triangle",
        ),
    ],
)
def test_region_occupancy(
    request: pytest.FixtureRequest,
    region_fixtures: list[str],
    data,
    expected_output: dict,
) -> None:
    """Tests region_occupancy for several RoIs.

    Checks whether the dimension, data, and coordinates of the computed
    occupancies are correct.
    """
    regions = [request.getfixturevalue(r) for r in region_fixtures]
    data = xr.DataArray(
        data=data,
        dims=["time", "space"],
        coords={"space": ["x", "y"]},
    )
    occupancies = compute_region_occupancy(data, regions)

    assert occupancies.dims == ("region", "time")
    assert (expected_output["data"] == occupancies.data).all()
    assert occupancies.region.values.tolist() == expected_output["coords"]


def test_region_occupancy_many_regions(
    triangle, unit_square, unit_square_with_hole, triangle_different_name
):
    """Tests occupancy for many RoIs with identical names.

    Ensures correct data and coordinate names for:
        - 1000 triangles suffixed _000 to _999
        - 100 unit squares suffixed _00 to _99
        - 10 unit squares with holes suffixed _0 to _9
        - 1 triangle named "pizza_slice" without suffix

    This test checks unique naming of coordinates in the computed
    occupancies when up to 1000 regions with identical names are passed,
    which is not covered in the other tests.
    """
    regions = (
        [triangle] * 1000
        + [unit_square] * 100
        + [unit_square_with_hole] * 10
        + [triangle_different_name] * 1
    )

    data = xr.DataArray(
        data=np.array([[0.5, 0.5]]),
        dims=["time", "space"],
        coords={"space": ["x", "y"]},
    )
    expected_output = xr.DataArray(
        data=np.array([[True]] * 1100 + [[False]] * 10 + [[True]] * 1),
        dims=["region", "time"],
        coords={
            "region": [f"triangle_{i:03d}" for i in range(1000)]
            + [f"Unit square_{i:02d}" for i in range(100)]
            + [f"Unit square with hole_{i:01d}" for i in range(10)]
            + ["pizza_slice"]
        },
    )
    occupancies = compute_region_occupancy(data, regions)
    xr.testing.assert_identical(occupancies, expected_output)


def test_region_occupancy_multiple_dims(triangle, two_individuals):
    """Tests region occupancy for data with common dimensions.

    This test ensures that the 'space' dimension is removed and the 'region'
    dimension is added, while all other dimensions ('time', 'keypoints',
    'individuals') are preserved.
    """
    regions = [triangle, triangle, triangle]
    occupancies = compute_region_occupancy(two_individuals, regions)

    input_dims = set(two_individuals.dims)
    output_dims = set(occupancies.dims)
    shared_dims = input_dims & output_dims

    assert shared_dims == {"time", "keypoints", "individuals"}
    assert input_dims - output_dims == {"space"}  # 'space' is removed
    assert output_dims - input_dims == {"region"}  # 'region' is added
    assert occupancies.region.shape == (len(regions),)


# ---------------------------------------------------------------------------
# Tests for compute_entry_exits
# ---------------------------------------------------------------------------


def _make_positions(coords_xy: list[tuple[float, float]]) -> xr.DataArray:
    """Build a simple (time, space) DataArray from a list of (x, y) tuples."""
    data = np.array(coords_xy, dtype=float)
    n = len(coords_xy)
    return xr.DataArray(
        data,
        dims=["time", "space"],
        coords={"space": ["x", "y"], "time": np.arange(n)},
    )


@pytest.mark.parametrize(
    "coords_xy, expected_events",
    [
        pytest.param(
            # starts outside, enters, stays, exits
            [(1.5, 1.5), (0.5, 0.5), (0.5, 0.5), (1.5, 1.5)],
            [0, 1, 0, -1],
            id="outside-inside-inside-outside",
        ),
        pytest.param(
            # starts inside immediately
            [(0.5, 0.5), (1.5, 1.5), (0.5, 0.5)],
            [1, -1, 1],
            id="starts-inside",
        ),
        pytest.param(
            # always outside — no events
            [(1.5, 1.5), (2.0, 2.0), (3.0, 3.0)],
            [0, 0, 0],
            id="always-outside",
        ),
        pytest.param(
            # always inside — only initial entry
            [(0.1, 0.1), (0.2, 0.2), (0.3, 0.3)],
            [1, 0, 0],
            id="always-inside",
        ),
    ],
)
def test_entry_exits_basic(unit_square, coords_xy, expected_events):
    """Test basic entry/exit detection with no keypoints dimension."""
    positions = _make_positions(coords_xy)
    events = compute_entry_exits(positions, [unit_square])

    assert events.dims == ("region", "time")
    assert events.sizes["time"] == len(coords_xy)
    np.testing.assert_array_equal(
        events.sel(region="Unit square").values,
        np.array(expected_events),
    )


def test_entry_exits_output_dims(unit_square, triangle):
    """Test that output has (region, time) dims and correct region coords."""
    positions = _make_positions([(0.5, 0.5), (1.5, 1.5), (0.5, 0.5)])
    events = compute_entry_exits(positions, [unit_square, triangle])

    assert events.dims == ("region", "time")
    assert events.sizes["region"] == 2
    assert list(events.region.values) == ["Unit square", "triangle"]


@pytest.mark.parametrize(
    "mode, expected_events",
    [
        pytest.param(
            # centroid = mean of kp0 + kp1 positions.
            # t0: (1.5,1.5) outside; t1: (1.0,1.0) boundary=inside;
            # t2: (0.5,0.5) inside; t3: (1.5,1.5) outside
            "centroid",
            [0, 1, 0, -1],
            id="centroid",
        ),
        pytest.param(
            # "all": both keypoints must be inside
            # t0: 0/2; t1: 1/2 (kp0 only); t2: 2/2; t3: 0/2
            "all",
            [0, 0, 1, -1],
            id="all",
        ),
        pytest.param(
            # "any": at least one keypoint inside
            # t0: 0; t1: kp0 in; t2: both in; t3: none
            "any",
            [0, 1, 0, -1],
            id="any",
        ),
        pytest.param(
            # "majority": sum > n_kp/2, i.e. sum > 1 for 2 keypoints
            # t0: 0>1=F; t1: 1>1=F; t2: 2>1=T; t3: 0>1=F
            "majority",
            [0, 0, 1, -1],
            id="majority",
        ),
    ],
)
def test_entry_exits_modes(unit_square, mode, expected_events):
    """Test keypoint aggregation modes.

    Two keypoints across 4 time steps:
    - t=0: kp0 outside, kp1 outside  -> no event for any mode
    - t=1: kp0 inside,  kp1 outside  -> depends on mode
    - t=2: kp0 inside,  kp1 inside   -> entry for all/majority
    - t=3: kp0 outside, kp1 outside  -> exit for all modes
    """
    # Build data with keypoints dim: shape (time, space, keypoints)
    # kp0 positions: outside at t0, inside at t1,t2, outside at t3
    # kp1 positions: outside at t0,t1, inside at t2, outside at t3
    kp0 = np.array(
        [[1.5, 1.5], [0.5, 0.5], [0.5, 0.5], [1.5, 1.5]],
        dtype=float,
    )
    kp1 = np.array(
        [[1.5, 1.5], [1.5, 1.5], [0.5, 0.5], [1.5, 1.5]],
        dtype=float,
    )
    # stack into (time, space, keypoints)
    data = np.stack([kp0, kp1], axis=-1)  # (4, 2, 2)
    positions = xr.DataArray(
        data,
        dims=["time", "space", "keypoints"],
        coords={
            "time": np.arange(4),
            "space": ["x", "y"],
            "keypoints": ["kp0", "kp1"],
        },
    )
    events = compute_entry_exits(positions, [unit_square], mode=mode)

    assert "keypoints" not in events.dims
    np.testing.assert_array_equal(
        events.sel(region="Unit square").values,
        np.array(expected_events),
    )


@pytest.mark.parametrize(
    "min_frames, expected_events",
    [
        pytest.param(
            1,
            # no filtering: every transition is captured
            [0, 1, -1, 1, 0, -1],
            id="min_frames=1 (no filter)",
        ),
        pytest.param(
            2,
            # brief single-frame inside at t=1 is suppressed;
            # sustained entry at t=3,4 fires at t=4;
            # single-frame exit at t=5 is also suppressed (debounce)
            [0, 0, 0, 0, 1, 0],
            id="min_frames=2",
        ),
    ],
)
def test_entry_exits_min_frames(unit_square, min_frames, expected_events):
    """Test that min_frames suppresses brief border detections."""
    # occupancy pattern: F T F T T F
    coords_xy = [
        (1.5, 1.5),  # outside
        (0.5, 0.5),  # inside  (brief — only 1 frame)
        (1.5, 1.5),  # outside
        (0.5, 0.5),  # inside
        (0.5, 0.5),  # inside  (sustained)
        (1.5, 1.5),  # outside
    ]
    positions = _make_positions(coords_xy)
    events = compute_entry_exits(
        positions, [unit_square], min_frames=min_frames
    )

    np.testing.assert_array_equal(
        events.sel(region="Unit square").values,
        np.array(expected_events),
    )


def test_entry_exits_starts_inside_with_min_frames(unit_square):
    """Test that starts-inside yields +1 at t=0 even when min_frames > 1.

    Validates the documented behaviour: ``time=0`` is always accepted
    as-is by the debounce logic, so a subject that begins inside the
    region records an entry at the very first frame regardless of
    ``min_frames``.  Also confirms that a brief single-frame exit is
    suppressed symmetrically.
    """
    # Occupancy pattern: T F T T
    # Brief exit at t=1 should be suppressed with min_frames=2.
    coords_xy = [
        (0.5, 0.5),  # inside  (initial)
        (1.5, 1.5),  # outside (brief — only 1 frame)
        (0.5, 0.5),  # inside  (sustained)
        (0.5, 0.5),  # inside
    ]
    positions = _make_positions(coords_xy)
    events = compute_entry_exits(positions, [unit_square], min_frames=2)

    # Only the initial entry at t=0 is recorded; brief exit suppressed
    np.testing.assert_array_equal(
        events.sel(region="Unit square").values,
        np.array([1, 0, 0, 0]),
    )


def test_entry_exits_preserves_individuals_dim(unit_square, two_individuals):
    """Test that the individuals dimension is preserved in output."""
    events = compute_entry_exits(two_individuals, [unit_square])

    assert "individuals" in events.dims
    assert "keypoints" not in events.dims
    assert "space" not in events.dims
    assert "region" in events.dims
    assert "time" in events.dims
