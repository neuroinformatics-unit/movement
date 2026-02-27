from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
import xarray as xr

from movement.kinematics.kinetic_energy import compute_kinetic_energy


@pytest.mark.parametrize("decompose", [True, False])
def test_basic_shape_and_values(decompose):
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
    result = compute_kinetic_energy(position, decompose=decompose)
    if decompose:
        assert set(result.dims) == {"time", "individuals", "energy"}
        assert list(result.coords["energy"].values) == [
            "translational",
            "internal",
        ]
        assert result.shape == (2, 1, 2)
    else:
        assert set(result.dims) == {"time", "individuals"}
        assert result.shape == (2, 1)
    assert (result >= 0).all()


def test_uniform_linear_motion(valid_poses_dataset):
    """Uniform rigid motion:
    expect translational energy > 0, internal ≈ 0.
    """
    ds = valid_poses_dataset.copy(deep=True)
    energy = compute_kinetic_energy(ds["position"], decompose=True)
    trans = energy.sel(energy="translational")
    internal = energy.sel(energy="internal")
    assert np.allclose(trans, 3)
    assert np.allclose(internal, 0)


@pytest.fixture
def spinning_dataset():
    """Create synthetic rotational-only dataset."""
    time = 10
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
    energy = compute_kinetic_energy(spinning_dataset, decompose=True)
    trans = energy.sel(energy="translational")
    internal = energy.sel(energy="internal")
    assert np.allclose(trans, 0)
    assert (internal > 0).all()


@pytest.mark.parametrize(
    "masses",
    [
        {"centroid": 2.0, "left": 2.0, "right": 2.0},
        {"centroid": 0.4, "left": 0.3, "right": 0.3},
    ],
)
def test_weighted_kinetic_energy(valid_poses_dataset, masses):
    """Kinetic energy should scale linearly with individual's total mass
    if velocity is constant.
    """
    ds = valid_poses_dataset.copy(deep=True)
    position = ds["position"]
    unweighted = compute_kinetic_energy(position)
    weighted = compute_kinetic_energy(position, masses=masses)
    factor = sum(masses.values()) / position.sizes["keypoints"]
    xr.testing.assert_allclose(weighted, unweighted * factor)


@pytest.mark.parametrize(
    "valid_poses_dataset, keypoints, expected_exception",
    [
        pytest.param(
            "multi_individual_array",
            None,
            does_not_raise(),
            id="3-keypoints (sufficient)",
        ),
        pytest.param(
            "multi_individual_array",
            ["centroid"],
            pytest.raises(ValueError, match="At least 2 keypoints"),
            id="3-keypoints 1-selected (insufficient)",
        ),
        pytest.param(
            "single_keypoint_array",
            None,
            pytest.raises(ValueError, match="At least 2 keypoints"),
            id="1-keypoint (insufficient)",
        ),
    ],
    indirect=["valid_poses_dataset"],
)
def test_insufficient_keypoints(
    valid_poses_dataset, keypoints, expected_exception
):
    """Function should raise error if fewer than 2 keypoints."""
    with expected_exception:
        compute_kinetic_energy(
            valid_poses_dataset["position"],
            keypoints=keypoints,
            decompose=True,
        )


@pytest.fixture
def position_array_with_nan():
    data = np.ones((5, 1, 2, 2))
    return xr.DataArray(
        data,
        coords={
            "time": np.arange(5),
            "individuals": ["ind1"],
            "keypoints": ["head", "tail"],
            "space": ["x", "y"],
        },
        dims=["time", "individuals", "keypoints", "space"],
    )


def test_kinetic_energy_partial_nan(position_array_with_nan):
    data = position_array_with_nan.copy(deep=True)
    data.loc[dict(time=2, keypoints="tail")] = np.nan
    ke = compute_kinetic_energy(data)
    assert np.isfinite(ke.sel(time=2, individuals="ind1"))


def test_kinetic_energy_full_nan(position_array_with_nan):
    data = position_array_with_nan.copy(deep=True)
    data.loc[dict(time=2)] = np.nan
    ke = compute_kinetic_energy(data)
    assert np.isnan(ke.sel(time=1, individuals="ind1"))
    assert np.isnan(ke.sel(time=3, individuals="ind1"))


def test_kinetic_energy_nan_warning(position_array_with_nan):
    data = position_array_with_nan.copy(deep=True)
    data.loc[dict(time=[1, 2], keypoints="head")] = np.nan
    with pytest.warns(UserWarning, match="The result may be unreliable"):
        compute_kinetic_energy(data, nan_warn_threshold=0.1)
