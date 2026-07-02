import re
from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
import xarray as xr

from movement.kinematics import (
    compute_directional_change,
    compute_path_deviation,
    compute_path_emax,
    compute_path_length,
    compute_path_sinuosity,
    compute_path_straightness,
    compute_turning_angle,
)

# Shared by all metrics with a minimum number of time points.
# The number itself is metric-specific, so it is left out of the match.
time_points_value_error = pytest.raises(
    ValueError,
    match="time points are required",
)

# Pre-sliced time-range cases shared by multiple unit tests
time_range_cases = [
    # full time ranges
    pytest.param(slice(None, None), does_not_raise(), id="full-range"),
    pytest.param(slice(0, None), does_not_raise(), id="from-zero"),
    pytest.param(slice(0, 9), does_not_raise(), id="explicit-full-range"),
    pytest.param(slice(0, 10), does_not_raise(), id="stop-beyond-data"),
    pytest.param(slice(-1, 9), does_not_raise(), id="start-before-data"),
    # partial time ranges
    pytest.param(slice(1, 8), does_not_raise(), id="partial-range"),
    pytest.param(slice(1.5, 8.5), does_not_raise(), id="fractional-bounds"),
    pytest.param(slice(2, None), does_not_raise(), id="from-two-to-end"),
    # Empty or too-short slices
    pytest.param(
        slice(9, 0), time_points_value_error, id="start-greater-than-stop"
    ),
    pytest.param(
        slice(0, 0.5), time_points_value_error, id="too-few-time-points"
    ),
]

# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────


@pytest.fixture
def valid_poses_dataset_with_cross_nan(valid_poses_dataset):
    """Return a valid poses dataset with NaNs crossing both dimensions.

    NaN layout (10 frames):
    - (right, id_0): 10/10 NaN (100%)
    - (centroid, id_1): 6/10 NaN (60%)
    - (right, id_1): 1/10 NaN (10%)
    - (centroid, id_0): 1/10 NaN (10%)
    - All other tracks: 0 NaN
    """
    position = valid_poses_dataset.position
    position.loc[{"individual": "id_0", "keypoint": "right"}] = np.nan
    position.loc[
        {
            "individual": "id_1",
            "keypoint": "centroid",
            "time": [0, 1, 2, 3, 4, 5],
        }
    ] = np.nan
    position.loc[{"individual": "id_1", "keypoint": "right", "time": 0}] = (
        np.nan
    )
    position.loc[{"individual": "id_0", "keypoint": "centroid", "time": 0}] = (
        np.nan
    )
    return valid_poses_dataset


@pytest.fixture
def straight_paths(valid_poses_dataset):
    """Return centroid position data from the valid poses dataset.

    Both individuals move in straight lines at constant steps of sqrt(2).
     "id_0" moves along the x=y line, while "id_1" moves along the x=-y line.
    """
    return valid_poses_dataset.position.sel(keypoint="centroid")


@pytest.fixture
def stationary_paths(straight_paths):
    """Return path data where the individuals never move."""
    path = straight_paths.copy()
    path[:] = 3.0
    return path


@pytest.fixture
def closed_loop_paths(straight_paths):
    """Return path data forming a closed loop.

    Overwrites ``straight_paths`` to end at starting point.
    """
    path = straight_paths.copy()
    path.loc[{"time": 9}] = path.loc[{"time": 0}]
    return path


@pytest.fixture
def sharp_turn_paths(straight_paths):
    """Return path data where both individuals take a sharp turn.

    Modifies the ``straight_paths`` fixture.
    After the first 9 frames (8 steps) of straight motion, both individuals
    drop straight back to the x-axis (y=0) for the 10th frame (last step).

    Path length: 8 segments of sqrt(2) + 1 segment of 8 = 8 * sqrt(2) + 8
    Straight-line distance: 8
    Straightness index: (8 / (8 * sqrt(2) + 8)) = 1 / (1 + sqrt(2))
    """
    path = straight_paths.copy()
    # Maintain x position for last step
    path.loc[{"time": 9, "space": "x"}] = path.sel(time=8, space="x")
    # Drop y position to 0 for last step
    path.loc[{"time": 9, "space": "y"}] = 0.0
    return path


@pytest.fixture
def straight_paths_3d(straight_paths):
    """Return straight paths extended to 3D with z=0."""
    return (
        straight_paths.pad(space=(0, 1))
        .fillna(0)
        .assign_coords(space=["x", "y", "z"])
    )


@pytest.fixture
def straight_paths_with_nan(straight_paths):
    """Return straight paths with NaN injected at time=5."""
    path = straight_paths.copy()
    path.loc[{"time": 5}] = np.nan
    return path


@pytest.fixture
def lateral_detour_paths(straight_paths):
    """Return path data where each individual takes a 1-unit lateral detour.

    id_0 moves along x=y (chord unit: (1/sqrt(2), 1/sqrt(2))).
    Its perpendicular is (-1/sqrt(2), +1/sqrt(2)).

    id_1 moves along x=-y (chord unit: (1/sqrt(2), -1/sqrt(2))).
    Its perpendicular is (1/sqrt(2), +1/sqrt(2)).

    At time=5, each individual is displaced exactly 1 unit perpendicular
    to its own chord. Expected deviation at time=5 is 1.0 for both.
    """
    path = straight_paths.copy()
    perp = 1.0 / np.sqrt(2)
    # id_0: perpendicular to x=y is (-1/sqrt(2), +1/sqrt(2))
    path.loc[{"time": 5, "space": "x", "individual": "id_0"}] -= perp
    path.loc[{"time": 5, "space": "y", "individual": "id_0"}] += perp
    # id_1: perpendicular to x=-y is (+1/sqrt(2), +1/sqrt(2))
    path.loc[{"time": 5, "space": "x", "individual": "id_1"}] += perp
    path.loc[{"time": 5, "space": "y", "individual": "id_1"}] += perp
    return path


@pytest.fixture
def regular_zigzag_path():
    """Unit-step 90-degree zigzag (East, North, East, North, ...).

    Every step has length 1, so the mean step length is 1 and the
    coefficient of variation of step length is 0. Every turn is 90
    degrees, so the mean cosine of the turning angles is
    cos(90 deg) = 0. Benhamou 2004 Eq. 8 (the one used to compute
    sinuosity) then reduces to
    S = 2 * [1 * (1 + 0) / (1 - 0)]^(-1/2) = 2 / sqrt(1) = 2.0
    """
    n = 20
    xy = np.zeros((n, 2))
    for i in range(1, n):
        xy[i] = xy[i - 1] + ([1, 0] if i % 2 else [0, 1])
    return xr.DataArray(
        xy,
        dims=["time", "space"],
        coords={"time": np.arange(n), "space": ["x", "y"]},
    )


@pytest.fixture
def regular_hexagon_path():
    """Return a path tracing a regular hexagon with step length 2.

    Every step turns by a constant 60 degrees, so the mean cosine of the
    turning angles is cos(60 deg) = 0.5, giving E_max-a = 0.5 / (1 - 0.5)
    = 1.0. With a constant step length of 2, E_max-b = 2.0.
    """
    r3 = np.sqrt(3)
    positions = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [3.0, r3],
            [2.0, 2 * r3],
            [0.0, 2 * r3],
            [-1.0, r3],
            [0.0, 0.0],
        ]
    )
    return xr.DataArray(
        positions,
        dims=["time", "space"],
        coords={"time": np.arange(len(positions)), "space": ["x", "y"]},
    )


@pytest.fixture
def scaled_zigzag_path(regular_zigzag_path):
    """Return the zigzag path with all positions scaled by a factor of 4.

    Sinuosity has units of 1/sqrt(length), so scaling the positions by
    k leaves the shape unchanged but divides S by sqrt(k).
    Here S = 2.0 / sqrt(4) = 1.0.
    """
    return regular_zigzag_path * 4


@pytest.fixture
def zigzag_path_with_nan(regular_zigzag_path):
    """Return the zigzag path with the position at time=10 missing.

    The missing position invalidates 2 step lengths and 3 turning angles,
    but every remaining step still has length 1 and every remaining turn
    is still 90 degrees, so sinuosity (S) is unchanged at 2.0.
    """
    path = regular_zigzag_path.copy()
    path.loc[{"time": 10}] = np.nan
    return path


@pytest.fixture
def regular_triangle_path():
    """Return a path tracing a regular triangle with unit step length.

    Every step turns by a constant 120 degrees, so the mean cosine of the
    turning angles is cos(120 deg) = -0.5, giving E_max-a = -0.5 / 1.5 =
    -1/3. This exercises the negative part of the E_max range.
    """
    r3 = np.sqrt(3)
    positions = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, r3 / 2],
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, r3 / 2],
            [0.0, 0.0],
        ]
    )
    return xr.DataArray(
        positions,
        dims=["time", "space"],
        coords={"time": np.arange(len(positions)), "space": ["x", "y"]},
    )


# ─────────────────────────────────────────────
# Cross-metric time-range tests
# ─────────────────────────────────────────────


@pytest.mark.parametrize("time_slice, expected_exception", time_range_cases)
@pytest.mark.parametrize(
    "compute_fn",
    [
        pytest.param(compute_path_straightness, id="straightness"),
        pytest.param(compute_directional_change, id="directional-change"),
        pytest.param(compute_path_deviation, id="deviation"),
        pytest.param(compute_path_sinuosity, id="sinuosity"),
        pytest.param(compute_path_emax, id="emax"),
    ],
)
def test_path_metrics_across_time_ranges(
    straight_paths,
    time_slice,
    expected_exception,
    compute_fn,
):
    """Test that path metrics accept valid time ranges and raise
    on too-few time points. Path length is tested separately because
    its time-range test also validates computed values.
    """
    position = straight_paths.sel(time=time_slice)
    with expected_exception:
        compute_fn(position)


# ─────────────────────────────────────────────
# Path length tests
# ─────────────────────────────────────────────


@pytest.mark.parametrize("time_slice, expected_exception", time_range_cases)
def test_path_length_across_time_ranges(
    valid_poses_dataset, time_slice, expected_exception
):
    """Test path length computation for a uniform linear motion case,
    across different pre-sliced time ranges.

    The test dataset ``valid_poses_dataset``
    contains 2 individuals ("id_0" and "id_1"), moving
    along x=y and x=-y lines, respectively, at a constant velocity.
    At each frame they cover a distance of sqrt(2) in x-y space, so in total
    we expect a path length of sqrt(2) * num_segments, where num_segments is
    the number of selected frames minus 1.
    """
    position = valid_poses_dataset.position.sel(time=time_slice)
    with expected_exception:
        path_length = compute_path_length(position)
        assert path_length.name == "path_length"
        assert path_length.long_name == "Path Length"

        num_segments = position.sizes["time"] - 1
        expected_path_length = xr.DataArray(
            np.ones((3, 2)) * np.sqrt(2) * num_segments,
            dims=["keypoint", "individual"],
            coords={
                "keypoint": position.coords["keypoint"],
                "individual": position.coords["individual"],
            },
        )
        xr.testing.assert_allclose(path_length, expected_path_length)


@pytest.mark.parametrize(
    "nan_policy, expected_path_lengths_id_0, expected_exception",
    [
        (
            "ffill",
            np.array([np.sqrt(2) * 9, np.sqrt(2) * 8, np.nan]),
            does_not_raise(),
        ),
        (
            "scale",
            np.array([np.sqrt(2) * 9, np.sqrt(2) * 9, np.nan]),
            does_not_raise(),
        ),
        (
            "invalid",  # invalid value for nan_policy
            np.zeros(3),
            pytest.raises(ValueError, match="Invalid value for nan_policy"),
        ),
    ],
)
def test_path_length_with_nan(
    valid_poses_dataset_with_nan,
    nan_policy,
    expected_path_lengths_id_0,
    expected_exception,
):
    """Test path length computation for a uniform linear motion case,
    with varying number of missing values per individual and keypoint.
    Because the underlying motion is uniform linear, the "scale" policy should
    perfectly restore the path length for individual "id_0" to its true value.
    The "ffill" policy should do likewise if frames are missing in the middle,
    but will not "correct" for missing values at the edges.
    """
    position = valid_poses_dataset_with_nan.position
    with (
        pytest.warns(UserWarning, match="The result may be unreliable"),
        expected_exception,
    ):
        path_length = compute_path_length(position, nan_policy=nan_policy)
        assert path_length.name == "path_length"
        # Get path_length for individual "id_0" as a numpy array
        path_length_id_0 = path_length.sel(individual="id_0").values
        # Check them against the expected values
        np.testing.assert_allclose(
            path_length_id_0, expected_path_lengths_id_0
        )


# Regex patterns to match the warning messages
_exclude_id_1_and_left = r"(?s)(?!.*id_1)(?!.*left)"
_include_threshold_100 = (
    r".*The result may be unreliable.*right.*id_0.*10/10.*"
)
_include_threshold_20 = (
    r".*The result may be unreliable"
    r".*centroid.*right.*id_0.*3/10.*10/10.*"
)
_cross_nan_threshold_50 = (
    r"(?s)(?!.*1/10)(?=.*10/10)(?=.*6/10)"
    r".*The result may be unreliable.*"
)


@pytest.mark.parametrize(
    "fixture_name, nan_warn_threshold, expected_exception",
    [
        pytest.param(
            "valid_poses_dataset_with_nan",
            1,
            pytest.warns(
                UserWarning,
                match=(f"{_exclude_id_1_and_left}{_include_threshold_100}"),
            ),
            id="standard-threshold-100",
        ),
        pytest.param(
            "valid_poses_dataset_with_nan",
            0.2,
            pytest.warns(
                UserWarning,
                match=(f"{_exclude_id_1_and_left}{_include_threshold_20}"),
            ),
            id="standard-threshold-20",
        ),
        pytest.param(
            "valid_poses_dataset_with_nan",
            -1,
            pytest.raises(ValueError, match="between 0 and 1"),
            id="invalid-threshold",
        ),
        pytest.param(
            "valid_poses_dataset_with_cross_nan",
            0.5,
            pytest.warns(
                UserWarning,
                match=_cross_nan_threshold_50,
            ),
            id="cross-nan-threshold-50",
        ),
    ],
)
def test_path_length_nan_warn_threshold(
    request,
    fixture_name,
    nan_warn_threshold,
    expected_exception,
):
    """Test that a warning is raised with matching message containing
    information on the individuals and keypoints whose number of missing
    values exceeds the given threshold or that an error is raised
    when the threshold is invalid.
    """
    dataset = request.getfixturevalue(fixture_name)
    position = dataset.position
    with expected_exception:
        result = compute_path_length(
            position, nan_warn_threshold=nan_warn_threshold
        )
        assert result.name == "path_length"


# ─────────────────────────────────────────────
# Path straightness tests
# ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "fixture_name, expected_value",
    [
        pytest.param(
            "straight_paths",
            1.0,
            id="straight-line",
        ),
        pytest.param(
            "closed_loop_paths",
            0.0,
            id="closed-loop",
        ),
        pytest.param(
            "sharp_turn_paths",
            1 / (1 + np.sqrt(2)),
            id="sharp-turn",
        ),
        pytest.param(
            "stationary_paths",
            np.nan,
            id="stationary",
        ),
    ],
)
@pytest.mark.parametrize("nan_policy", ["ffill", "scale"])
def test_path_straightness_known_values(
    request, fixture_name, expected_value, nan_policy
):
    """Test that the straightness index matches expected values
    for trajectories with known geometry, regardless of NaN handling policy.
    """
    position = request.getfixturevalue(fixture_name)
    result = compute_path_straightness(position, nan_policy=nan_policy)
    if np.isnan(expected_value):
        assert result.isnull().all()
    else:
        xr.testing.assert_allclose(
            result,
            xr.full_like(result, expected_value),
        )


# ─────────────────────────────────────────────
# Turning angle tests
# ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "in_degrees, expected_units",
    [(True, "degrees"), (False, "radians")],
)
def test_turning_angle_output_shape_and_attributes(
    valid_poses_dataset, in_degrees, expected_units
):
    """Test that the function returns the correct shape,
    dimensions, and attributes.
    """
    position = valid_poses_dataset.position
    angles = compute_turning_angle(position, in_degrees=in_degrees)

    # Space dimension must be dropped, others preserved
    assert angles.sizes["time"] == position.sizes["time"]
    assert "space" not in angles.dims
    assert "individual" in angles.dims

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
def test_turning_angle_with_invalid_input(
    invalid_data, expected_error, expected_match_str
):
    """Test that invalid inputs raise the expected error."""
    with pytest.raises(expected_error, match=re.escape(expected_match_str)):
        compute_turning_angle(invalid_data)


@pytest.mark.parametrize(
    "positions, expected_angle_deg",
    [
        pytest.param(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
            0.0,
            id="straight_line",
        ),
        pytest.param(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
            90.0,
            id="pos_turn_90",
        ),
        pytest.param(
            [[0.0, 0.0], [1.0, 0.0], [1.0, -1.0]],
            -90.0,
            id="neg_turn_90",
        ),
        pytest.param(
            [
                [0.0, 0.0],
                [
                    np.cos(np.deg2rad(170)),
                    np.sin(np.deg2rad(170)),
                ],
                [
                    np.cos(np.deg2rad(170)) + np.cos(np.deg2rad(-170)),
                    np.sin(np.deg2rad(170)) + np.sin(np.deg2rad(-170)),
                ],
            ],
            20.0,
            id="wrap_across_pi_boundary",
        ),
        pytest.param(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]],
            180.0,
            id="u_turn_180",
        ),
    ],
)
def test_turning_angle_known_values(positions, expected_angle_deg):
    """Test mathematical correctness of
    turning angles for specific trajectories.
    """
    pos_array = np.array(positions)
    data = xr.DataArray(
        pos_array,
        dims=["time", "space"],
        coords={
            "time": np.arange(len(pos_array)),
            "space": ["x", "y"],
        },
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
def test_turning_angle_min_step_length_masking(min_step, expect_nan):
    """Test that steps smaller than min_step_length
    result in NaN turning angles.
    """
    positions = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0 + 1e-5, 1e-5], [2.0, 1e-5]]
    )
    data = xr.DataArray(
        positions,
        dims=["time", "space"],
        coords={
            "time": np.arange(len(positions)),
            "space": ["x", "y"],
        },
    )

    angles = compute_turning_angle(data, min_step_length=min_step)

    assert np.isnan(angles.isel(time=2).item()) == expect_nan
    assert np.isnan(angles.isel(time=3).item()) == expect_nan


@pytest.mark.parametrize(
    "positions",
    [
        pytest.param(
            [[5.0, 5.0], [5.0, 5.0], [5.0, 5.0], [5.0, 5.0]],
            id="stationary",
        ),
        pytest.param(
            [[0.0, 0.0], [1.0, 0.0]],
            id="only_two_timepoints",
        ),
    ],
)
def test_turning_angle_all_nan_output(positions):
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


def test_turning_angle_nan_propagation(valid_poses_dataset):
    """Test that a NaN position correctly
    invalidates adjacent turning angles.
    """
    position = valid_poses_dataset.position.astype(float)

    position.loc[{"time": 2, "individual": "id_0", "keypoint": "left"}] = (
        np.nan
    )  # type: ignore[index]

    angles = compute_turning_angle(position)

    # A NaN at t=2 must break the steps at t=2 and t=3
    assert np.isnan(
        angles.sel(time=2, individual="id_0", keypoint="left").item()
    )
    assert np.isnan(
        angles.sel(time=3, individual="id_0", keypoint="left").item()
    )


def test_turning_angle_stationary_keypoint_independent_masking():
    """Zero-step masking is per-keypoint:
    moving kp has valid angles; stationary kp all NaN.
    """
    data = np.zeros((4, 2, 2))

    # kp_0: moves along x-axis (0, 1, 2, 3)
    data[:, 0, 0] = [0, 1, 2, 3]
    # kp_1: stays completely stationary at (5.0, 5.0)
    data[:, 1, :] = 5.0

    ds = xr.DataArray(
        data,
        dims=["time", "keypoint", "space"],
        coords={
            "time": np.arange(4),
            "keypoint": ["kp_0", "kp_1"],
            "space": ["x", "y"],
        },
    )

    angles = compute_turning_angle(ds)

    # Moving keypoint (kp_0): NaN at t=0, t=1;
    # valid (0.0) at t=2, t=3
    angles_kp0 = angles.isel(keypoint=0)
    assert np.isnan(angles_kp0.isel(time=0).item())
    assert np.isnan(angles_kp0.isel(time=1).item())
    assert np.allclose(
        angles_kp0.isel(time=slice(2, None)).values,
        0.0,
        atol=1e-10,
    )

    # Stationary keypoint (kp_1): should be all NaN
    angles_kp1 = angles.isel(keypoint=1)
    assert np.all(np.isnan(angles_kp1.values))


# ─────────────────────────────────────────────
# Directional change tests
# ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "fixture_name, expected_value, expected_all_nan",
    [
        pytest.param("straight_paths", 0.0, False, id="straight-line"),
        pytest.param("stationary_paths", None, True, id="stationary"),
    ],
)
def test_directional_change_known_values(
    request, fixture_name, expected_value, expected_all_nan
):
    """Test directional change for trajectories with known geometry.

    Straight-line motion produces zero turning angle, so DC is 0 at
    every valid time step. Stationary paths produce NaN turning angles,
    so DC is NaN everywhere.
    """
    position = request.getfixturevalue(fixture_name)
    dc = compute_directional_change(position)
    assert dc.name == "directional_change"
    assert dc.attrs["long_name"] == "Directional Change"

    if expected_all_nan:
        assert dc.isnull().all()
    else:
        valid = dc.isel(time=slice(2, None))
        xr.testing.assert_allclose(valid, xr.full_like(valid, expected_value))
        # Only the first two time steps are NaN.
        assert dc.isel(time=[0, 1]).isnull().all()


@pytest.mark.parametrize(
    "in_degrees, expected_turn",
    [
        pytest.param(False, 3 * np.pi / 4, id="radians"),
        pytest.param(True, 135.0, id="degrees"),
    ],
)
def test_directional_change_nonuniform_time(
    sharp_turn_paths, in_degrees, expected_turn
):
    """Test DC against a known nonzero value on non-uniform time.

    In ``sharp_turn_paths`` both individuals move in a straight line for
    8 steps, then turn sharply on the final step, producing a turning
    angle of ``3 * pi / 4`` (135 degrees) at the last time step.
    Dividing by the turning interval ``t[-1] - t[-3]`` gives the
    expected DC there. The non-uniform ``time`` coordinates ensure the
    interval is aligned to the turning angle's support (positions
    ``i-2..i``) rather than a centered difference around ``i``.
    """
    time = np.array([0, 1, 2, 4, 7, 11, 16, 22, 29, 37], dtype=float)
    path = sharp_turn_paths.assign_coords(time=time)

    dc = compute_directional_change(path, in_degrees=in_degrees)

    expected_last = expected_turn / (time[-1] - time[-3])
    last = dc.isel(time=-1)
    xr.testing.assert_allclose(last, xr.full_like(last, expected_last))
    # Only the first two time steps are NaN; the last step is now valid.
    assert dc.isel(time=[0, 1]).isnull().all()
    assert dc.isel(time=slice(2, None)).notnull().all()


# ─────────────────────────────────────────────
# Path deviation tests
# ─────────────────────────────────────────────

degenerate_chord_error = pytest.raises(
    ValueError,
    match=(
        "Path deviation is undefined because the start and end positions "
        "are identical for all tracks."
    ),
)


@pytest.mark.parametrize(
    "fixture_name", ["straight_paths", "straight_paths_3d"]
)
@pytest.mark.parametrize(
    # Should also hold true for partial slices of straight paths
    "time_slice",
    [slice(None), slice(2, 7)],
)
def test_path_deviation_straight_path_is_zero(
    request, fixture_name, time_slice
):
    """Deviation from a straight-line path is zero (2D and 3D).
    Also checks that the output has the expected attributes.
    """
    path = request.getfixturevalue(fixture_name).sel(time=time_slice)
    result = compute_path_deviation(path)
    xr.testing.assert_allclose(result, xr.zeros_like(result))

    assert result.name == "path_deviation"
    assert result.attrs["long_name"] == "Path Deviation"


def test_path_deviation_with_nan(straight_paths_with_nan):
    """Test that NaN positions result in NaN deviation at those time steps."""
    result = compute_path_deviation(straight_paths_with_nan)
    assert np.isnan(result.sel(time=5).values).all()
    off_nan = result.sel(time=[t for t in result.time.values if t != 5])
    xr.testing.assert_allclose(off_nan, xr.zeros_like(off_nan))


def test_path_deviation_known_value(lateral_detour_paths):
    """Deviation at the detour frame is 1.0 for both individuals;
    all other frames are 0.0.
    """
    result = compute_path_deviation(lateral_detour_paths)
    at_detour = result.sel(time=5)
    xr.testing.assert_allclose(at_detour, xr.full_like(at_detour, 1.0))
    off_detour = result.sel(time=[t for t in result.time.values if t != 5])
    xr.testing.assert_allclose(off_detour, xr.zeros_like(off_detour))


@pytest.mark.parametrize(
    "fixture_name", ["stationary_paths", "closed_loop_paths"]
)
def test_path_deviation_degenerate_chord_raises(request, fixture_name):
    """Test that a ValueError is raised when the start and end positions are
    identical for all tracks (deviation is undefined).
    """
    with degenerate_chord_error:
        compute_path_deviation(request.getfixturevalue(fixture_name))


def test_path_deviation_partially_degenerate_warns(straight_paths):
    """Test that a warning is raised when the start and end positions are
    identical for some but not all tracks, and that the result is NaN for
    the degenerate tracks and valid for the non-degenerate tracks.
    """
    path = straight_paths.copy()
    # Make id_0 stationary
    path.loc[{"individual": "id_0"}] = path.loc[
        {"individual": "id_0", "time": 0}
    ]

    with pytest.warns(
        UserWarning,
        match="Path deviation is undefined for tracks where the start and end",
    ):
        result = compute_path_deviation(path)
        assert np.isnan(result.sel(individual="id_0").values).all()
        id_1 = result.sel(individual="id_1")
        xr.testing.assert_allclose(id_1, xr.zeros_like(id_1))


# ─────────────────────────────────────────────
# Path sinuosity tests
# ─────────────────────────────────────────────


def test_path_sinuosity_too_few_timepoints(straight_paths):
    """Test that sinuosity enforces the minimum length (3) requirement."""
    # Slice the data to exactly 2 time points (which is only 1 segment)
    position = straight_paths.isel(time=slice(0, 2))
    with pytest.raises(ValueError, match="3 time points are required"):
        compute_path_sinuosity(position)


@pytest.mark.parametrize(
    "fixture_name, expected_value",
    [
        pytest.param("straight_paths", 0.0, id="straight-line"),
        pytest.param("stationary_paths", np.nan, id="stationary"),
        pytest.param("regular_zigzag_path", 2.0, id="zigzag"),
        pytest.param("scaled_zigzag_path", 1.0, id="zigzag-scaled-4x"),
        pytest.param("zigzag_path_with_nan", 2.0, id="zigzag-with-nan"),
    ],
)
def test_path_sinuosity_known_values(request, fixture_name, expected_value):
    """Test that sinuosity matches expected values for standard geometries."""
    position = request.getfixturevalue(fixture_name)
    result = compute_path_sinuosity(position)
    assert result.name == "sinuosity"
    assert result.long_name == "Path Sinuosity"

    assert set(result.dims) == set(position.dims) - {"time", "space"}
    if np.isnan(expected_value):
        assert result.isnull().all()
    else:
        xr.testing.assert_allclose(
            result,
            xr.full_like(result, expected_value),
            atol=1e-7,
        )


def test_path_sinuosity_raises_on_3d(straight_paths_3d):
    """Sinuosity is only defined for 2D data (inherited from turning angle)."""
    with pytest.raises(
        ValueError, match="Dimension 'space' must only contain"
    ):
        compute_path_sinuosity(straight_paths_3d)


def test_path_sinuosity_all_nan(straight_paths):
    """Test that fully missing tracks warn and yield NaN sinuosity."""
    position = straight_paths.copy()
    position[:] = np.nan
    with pytest.warns(UserWarning, match="The result may be unreliable"):
        result = compute_path_sinuosity(position)
    assert result.isnull().all()


def test_path_sinuosity_variable_reversals():
    """Biological edge case: back-and-forth pacing with variable step lengths.

    An animal repeatedly reversing direction but with naturalistic, variable
    step lengths. This is a highly tortuous movement and should compute
    successfully to a finite, positive value.
    """
    rng = np.random.default_rng(11)
    n = 30
    x = np.zeros(n)
    for i in range(1, n):
        # Reverse direction each step, but multiply by a random stride length
        x[i] = x[i - 1] + rng.choice([-1, 1]) * rng.uniform(0.5, 2.0)
    xy = np.column_stack([x, np.zeros(n)])

    data = xr.DataArray(
        xy,
        dims=["time", "space"],
        coords={"time": np.arange(n), "space": ["x", "y"]},
    )
    result = compute_path_sinuosity(data)

    assert not result.isnull().any()
    assert float(result.item()) > 0
    assert np.isfinite(result.item())


def test_path_sinuosity_outlier_step():
    """Artefact edge case: Tracking 'teleportation'.

    Simulates a tracker losing ID and assigning a point across the arena.
    The massive jump creates an extreme standard deviation in step length (b).
    The metric should not crash, but rather return a valid, finite float.
    """
    rng = np.random.default_rng(20)
    n = 50
    # Normal erratic movement
    xy = np.column_stack([np.arange(n, dtype=float), rng.normal(0, 0.3, n)])
    # Single massive tracking failure
    xy[25] = [1000.0, 1000.0]

    data = xr.DataArray(
        xy,
        dims=["time", "space"],
        coords={"time": np.arange(n), "space": ["x", "y"]},
    )
    result = compute_path_sinuosity(data)

    assert not result.isnull().any()
    assert np.isfinite(result.item())


# ─────────────────────────────────────────────
# E_max tests
# ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "fixture_name, in_spatial_units, expected_value",
    [
        pytest.param(
            "regular_hexagon_path", False, 1.0, id="hexagon-dimensionless"
        ),
        pytest.param(
            "regular_hexagon_path", True, 2.0, id="hexagon-spatial-units"
        ),
        pytest.param(
            "regular_triangle_path", False, -1 / 3, id="triangle-negative"
        ),
    ],
)
def test_path_emax_known_values(
    request, fixture_name, in_spatial_units, expected_value
):
    """Test E_max against paths with a constant turning angle.

    For a constant turning angle the mean cosine is exactly its cosine,
    so both E_max variants have closed-form values. The triangle case
    (120 degree turns) has a negative mean cosine, exercising the
    negative part of the E_max range.
    """
    path = request.getfixturevalue(fixture_name)
    result = compute_path_emax(path, in_spatial_units=in_spatial_units)
    assert result.name == "path_emax"
    assert result.attrs["long_name"] == "Maximum Expected Displacement"
    assert np.isclose(result.item(), expected_value)


@pytest.mark.parametrize("in_spatial_units", [False, True])
def test_path_emax_straight_path_is_inf(straight_paths, in_spatial_units):
    """A perfectly straight path has a mean cosine of 1, so E_max is +inf."""
    result = compute_path_emax(
        straight_paths, in_spatial_units=in_spatial_units
    )
    assert np.isinf(result).all()
    assert (result > 0).all()


@pytest.mark.parametrize("in_spatial_units", [False, True])
@pytest.mark.parametrize(
    "positions",
    [
        pytest.param(
            [[5.0, 5.0], [5.0, 5.0], [5.0, 5.0], [5.0, 5.0]],
            id="stationary",
        ),
        pytest.param(
            [[0.0, 0.0], [1.0, 0.0]],
            id="only_two_timepoints",
        ),
    ],
)
def test_path_emax_all_nan_output(positions, in_spatial_units):
    """Test cases where E_max should be NaN: a stationary track has no
    valid turning angles, and two time points define no turning angle.
    """
    data = xr.DataArray(
        np.array(positions),
        dims=["time", "space"],
        coords={
            "time": np.arange(len(positions)),
            "space": ["x", "y"],
        },
    )
    result = compute_path_emax(data, in_spatial_units=in_spatial_units)
    assert result.isnull().all()


def test_path_emax_output_shape(straight_paths):
    """Test that ``time`` and ``space`` are removed and other dimensions
    are preserved.
    """
    result = compute_path_emax(straight_paths)
    assert "space" not in result.dims
    assert "time" not in result.dims
    assert "individual" in result.dims


def test_path_emax_raises_on_3d(straight_paths_3d):
    """E_max is only defined for 2D data (inherited from turning angle)."""
    with pytest.raises(
        ValueError, match="Dimension 'space' must only contain"
    ):
        compute_path_emax(straight_paths_3d)
