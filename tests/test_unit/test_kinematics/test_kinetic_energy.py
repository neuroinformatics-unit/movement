import numpy as np
import pytest
import xarray as xr

from movement.kinematics.kinetic_energy import compute_kinetic_energy


def test_basic_shape_and_values():
    """Basic sanity check with simple data."""
    data = np.array(
        [[[[1, 0], [0, 1], [1, 1]]], [[[2, 0], [0, 2], [2, 2]]]]
    )  # shape: (time, individuals, keypoints, space)

    position = xr.DataArray(
        data,
        dims=["time", "individuals", "keypoints", "space"],
        coords={
            "time": [0, 1],
            "individuals": [0],
            "keypoints": [0, 1, 2],
            "space": ["x", "y"],
        },
    )

    result = compute_kinetic_energy(position)

    assert set(result.dims) == {"time", "individuals", "energy"}
    assert list(result.coords["energy"].values) == [
        "translational",
        "rotational",
    ]
    assert result.shape == (2, 1, 2)
    assert (result >= 0).all()


def test_uniform_linear_motion(valid_poses_dataset):
    """Uniform rigid motion:
    expect translational energy > 0, rotational ≈ 0.
    """
    ds = valid_poses_dataset.copy(deep=True)
    times = ds.coords["time"].values

    tx = times  # Linear motion along x
    ty = np.zeros_like(times)  # No motion in y

    offsets = {
        "centroid": [0, 0],
        "left": [-1, 0],
        "right": [1, 0],
    }

    for kp, (dx, dy) in offsets.items():
        for ind in ds.coords["individuals"].values:
            ds["position"].loc[
                dict(space="x", keypoints=kp, individuals=ind)
            ] = tx + dx
            ds["position"].loc[
                dict(space="y", keypoints=kp, individuals=ind)
            ] = ty + dy

    energy = compute_kinetic_energy(ds["position"])
    trans = energy.sel(energy="translational")
    rot = energy.sel(energy="rotational")

    assert (trans > 0).all()
    assert np.allclose(rot, 0, atol=1.1)


@pytest.fixture
def spinning_dataset():
    """Create synthetic rotational-only dataset."""
    time = 3
    keypoints = 4
    angles = np.linspace(0, 2 * np.pi, time)
    radius = 1.0

    positions = []
    for theta in angles:
        snapshot = []
        for k in range(keypoints):
            angle = theta + k * np.pi / 2
            snapshot.append([radius * np.cos(angle), radius * np.sin(angle)])
        positions.append([snapshot])  # 1 individual

    return xr.DataArray(
        np.array(positions),
        dims=["time", "individuals", "keypoints", "space"],
        coords={
            "time": np.arange(time),
            "individuals": ["id0"],
            "keypoints": [f"k{i}" for i in range(keypoints)],
            "space": ["x", "y"],
        },
    )


def test_pure_rotation(spinning_dataset):
    """In pure rotational motion, translational energy ≈ 0."""
    energy = compute_kinetic_energy(spinning_dataset)
    trans = energy.sel(energy="translational")
    rot = energy.sel(energy="rotational")

    assert np.allclose(trans, 0, atol=1e-6)
    assert (rot > 0).all()


def test_weighted_kinetic_energy(valid_poses_dataset):
    """Kinetic energy scales linearly with mass if velocity is constant."""
    ds = valid_poses_dataset.copy(deep=True)
    times = ds.coords["time"].values

    tx = times
    ty = times * 2

    offsets = {
        "centroid": [0, 0],
        "left": [-1, 0],
        "right": [1, 0],
    }

    for kp, (dx, dy) in offsets.items():
        for ind in ds.coords["individuals"].values:
            ds["position"].loc[
                dict(space="x", keypoints=kp, individuals=ind)
            ] = tx + dx
            ds["position"].loc[
                dict(space="y", keypoints=kp, individuals=ind)
            ] = ty + dy

    position = ds["position"]
    masses = {"centroid": 2.0, "left": 2.0, "right": 2.0}

    unweighted = compute_kinetic_energy(position)
    weighted = compute_kinetic_energy(position, masses=masses)

    assert (
        weighted.sel(energy="translational")
        >= unweighted.sel(energy="translational")
    ).all()
    assert (
        weighted.sel(energy="rotational")
        >= unweighted.sel(energy="rotational")
    ).all()


@pytest.mark.parametrize(
    "valid_poses_dataset", ["single_keypoint_array"], indirect=True
)
def test_insufficient_keypoints(valid_poses_dataset):
    """Function should raise error if fewer than 2 keypoints."""
    with pytest.raises(
        ValueError, match="Position array must contain at least two keypoints"
    ):
        compute_kinetic_energy(valid_poses_dataset["position"])
