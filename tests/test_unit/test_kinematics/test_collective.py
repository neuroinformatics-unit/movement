import numpy as np
import pytest
import xarray as xr

from movement.kinematics.collective import compute_group_spread


def test_compute_group_spread_with_known_values():
    """Test group spread on a simple dataset with known outputs."""
    position = xr.DataArray(
        np.array(
            [
                [[0.0, 2.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 4.0]],
            ]
        ),
        dims=["time", "space", "individuals"],
        coords={
            "time": [0, 1],
            "space": ["x", "y"],
            "individuals": ["id_0", "id_1"],
        },
    )

    result = compute_group_spread(position)

    expected = xr.DataArray(
        np.array([1.0, 2.0]),
        dims=["time"],
        coords={"time": [0, 1]},
        name="group_spread",
    )
    xr.testing.assert_equal(result, expected)


def test_compute_group_spread_prefers_centroid(valid_poses_dataset):
    """Test automatic keypoint selection when ``centroid`` is present."""
    result = compute_group_spread(valid_poses_dataset.position)

    expected = xr.DataArray(
        np.arange(valid_poses_dataset.sizes["time"], dtype=float),
        dims=["time"],
        coords={"time": valid_poses_dataset.time.values},
        name="group_spread",
    )
    xr.testing.assert_equal(result, expected)


def test_compute_group_spread_with_explicit_keypoint(valid_poses_dataset):
    """Test spread computation for an explicitly selected keypoint."""
    result = compute_group_spread(
        valid_poses_dataset.position, keypoint="left"
    )

    time = valid_poses_dataset.time.values.astype(float)
    expected = xr.DataArray(
        np.sqrt(time**2 + time + 0.5),
        dims=["time"],
        coords={"time": valid_poses_dataset.time.values},
        name="group_spread",
    )
    xr.testing.assert_allclose(result, expected)


def test_compute_group_spread_with_single_keypoint(valid_poses_dataset):
    """Test that a single available keypoint is selected automatically."""
    single_keypoint_position = valid_poses_dataset.position.sel(
        keypoints=["left"]
    )

    result = compute_group_spread(single_keypoint_position)
    expected = compute_group_spread(
        valid_poses_dataset.position, keypoint="left"
    )

    xr.testing.assert_equal(result, expected)


def test_compute_group_spread_ignores_nans():
    """Test NaN-safe behavior when some individuals are missing."""
    position = xr.DataArray(
        np.array(
            [
                [[0.0, 2.0, np.nan], [0.0, 0.0, np.nan]],
                [[0.0, 4.0, 2.0], [0.0, 0.0, 0.0]],
            ]
        ),
        dims=["time", "space", "individuals"],
        coords={
            "time": [0, 1],
            "space": ["x", "y"],
            "individuals": ["id_0", "id_1", "id_2"],
        },
    )

    result = compute_group_spread(position)

    expected = xr.DataArray(
        np.array([1.0, np.sqrt(8.0 / 3.0)]),
        dims=["time"],
        coords={"time": [0, 1]},
        name="group_spread",
    )
    xr.testing.assert_allclose(result, expected)


def test_compute_group_spread_without_individuals_dimension():
    """Test that missing ``individuals`` raises a clear error."""
    position = xr.DataArray(
        np.zeros((2, 2)),
        dims=["time", "space"],
        coords={"time": [0, 1], "space": ["x", "y"]},
    )

    with pytest.raises(
        ValueError, match="must have an 'individuals' dimension"
    ):
        compute_group_spread(position)


def test_compute_group_spread_with_multiple_keypoints_without_selection():
    """Test that ambiguous keypoint input raises a clear error."""
    position = xr.DataArray(
        np.zeros((2, 2, 2, 2)),
        dims=["time", "space", "keypoints", "individuals"],
        coords={
            "time": [0, 1],
            "space": ["x", "y"],
            "keypoints": ["left", "right"],
            "individuals": ["id_0", "id_1"],
        },
    )

    with pytest.raises(
        ValueError,
        match="Multiple keypoints present; pass `keypoint` to select one",
    ):
        compute_group_spread(position)
