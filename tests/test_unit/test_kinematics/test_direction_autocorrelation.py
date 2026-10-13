"""Tests for spatial direction autocorrelation."""

import numpy as np
import pytest
import xarray as xr

from movement.kinematics import compute_direction_autocorrelation


def position(x, y=None):
    """Make a two-dimensional position track from x and y coordinates."""
    x = np.asarray(x, dtype=float)
    if y is None:
        y = np.zeros_like(x)
    return xr.DataArray(
        np.column_stack((x, y)),
        dims=("time", "space"),
        coords={"time": np.arange(len(x)), "space": ["x", "y"]},
    )


def test_straight_path_and_spatial_lag():
    track = position([0, 1, 4, 5, 6])

    result = compute_direction_autocorrelation(
        track, step_length=1, delta_s_max=6
    )

    assert result.name == "direction_autocorrelation"
    np.testing.assert_array_equal(result.lag, [1, 2, 3, 4, 5, 6])
    np.testing.assert_allclose(result.values[:5], 1)
    assert np.isnan(result.values[5])


def test_alternating_directions():
    track = position([0, 1, 0, 1, 0, 1])

    result = compute_direction_autocorrelation(
        track, step_length=1, delta_s_max=3
    )

    np.testing.assert_allclose(result.values, [-1, 1, -1])


def test_same_path_with_different_temporal_sampling():
    sparse = position([0, 1, 2, 2, 2], [0, 0, 0, 1, 2])
    dense = position(
        [0, 0.5, 1, 1.5, 2, 2, 2, 2, 2],
        [0, 0, 0, 0, 0, 0.5, 1, 1.5, 2],
    )

    sparse_result = compute_direction_autocorrelation(
        sparse, step_length=1, delta_s_max=3
    )
    dense_result = compute_direction_autocorrelation(
        dense, step_length=1, delta_s_max=3
    )

    xr.testing.assert_allclose(sparse_result, dense_result)
    np.testing.assert_allclose(sparse_result.values[:2], [2 / 3, 0])


def test_missing_positions_split_segments():
    track = position([0, 1, 2, np.nan, 100, 101, 102])

    result = compute_direction_autocorrelation(
        track, step_length=1, delta_s_max=2
    )

    assert result.sel(lag=1).item() == 1
    assert np.isnan(result.sel(lag=2).item())


def test_stationary_positions_have_no_heading():
    track = position([0, 0, 0, 0])

    result = compute_direction_autocorrelation(
        track, step_length=1, delta_s_max=2
    )

    assert np.isnan(result).all()
    with pytest.raises(ValueError, match="Cannot infer step_length"):
        compute_direction_autocorrelation(track)


def test_repeated_positions_do_not_create_headings():
    repeated = position([0, 0, 1, 1, 2, 2, 3])
    regular = position([0, 1, 2, 3])

    result = compute_direction_autocorrelation(
        repeated, step_length=1, delta_s_max=2
    )
    expected = compute_direction_autocorrelation(
        regular, step_length=1, delta_s_max=2
    )

    xr.testing.assert_allclose(result, expected)


def test_other_dimensions_are_preserved():
    first = position([0, 1, 2, 3, 4])
    second = position([4, 3, 4, 3, 4])
    tracks = xr.concat(
        [first, second], dim=xr.IndexVariable("individual", ["a", "b"])
    ).expand_dims(keypoint=["centroid"])

    result = compute_direction_autocorrelation(
        tracks, step_length=1, delta_s_max=2
    )

    assert set(result.dims) == {"individual", "keypoint", "lag"}
    np.testing.assert_allclose(
        result.sel(individual="a", keypoint="centroid").values,
        [1, 1],
    )
    np.testing.assert_allclose(
        result.sel(individual="b", keypoint="centroid").values,
        [-1, 1],
    )


def test_chunked_track():
    pytest.importorskip("dask")
    track = position([0, 1, 2, 3, 4]).chunk({"time": 2})

    result = compute_direction_autocorrelation(track, delta_s_max=2).compute()

    np.testing.assert_allclose(result.values, [1, 1])


@pytest.mark.parametrize("step_length", [0, -1, np.inf, np.nan, True])
def test_invalid_step_length(step_length):
    track = position([0, 1, 2])
    with pytest.raises(ValueError, match="step_length"):
        compute_direction_autocorrelation(track, step_length=step_length)


@pytest.mark.parametrize("delta_s_max", [0, -1, 1.5, True])
def test_invalid_delta_s_max(delta_s_max):
    track = position([0, 1, 2])
    with pytest.raises(ValueError, match="delta_s_max"):
        compute_direction_autocorrelation(
            track, step_length=1, delta_s_max=delta_s_max
        )
