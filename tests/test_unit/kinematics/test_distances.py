"""Tests for distance-related kinematic functions."""

import numpy as np
import pytest
import xarray as xr

from movement.kinematics.distances import _cdist, compute_pairwise_distances


@pytest.mark.parametrize(
    "dim, expected_data",
    [
        (
            "individuals",
            np.array(
                [
                    [
                        [0.0, 1.0, 1.0],
                        [1.0, np.sqrt(2), 0.0],
                        [1.0, 2.0, np.sqrt(2)],
                    ],
                    [
                        [2.0, np.sqrt(5), 1.0],
                        [3.0, np.sqrt(10), 2.0],
                        [np.sqrt(5), np.sqrt(8), np.sqrt(2)],
                    ],
                ]
            ),
        ),
        (
            "keypoints",
            np.array(
                [[[1.0, 1.0], [1.0, 1.0]], [[1.0, np.sqrt(5)], [3.0, 1.0]]]
            ),
        ),
    ],
)
def test_cdist_with_known_values(dim, expected_data, valid_poses_dataset):
    labels_dim = "keypoints" if dim == "individuals" else "individuals"
    input_dataarray = valid_poses_dataset.position.sel(time=slice(0, 1))
    pairs = input_dataarray[dim].values[:2]
    expected = xr.DataArray(
        expected_data,
        coords=[
            input_dataarray.time.values,
            getattr(input_dataarray, labels_dim).values,
            getattr(input_dataarray, labels_dim).values,
        ],
        dims=["time", pairs[0], pairs[1]],
    )
    a = input_dataarray.sel({dim: pairs[0]})
    b = input_dataarray.sel({dim: pairs[1]})
    result = _cdist(a, b, dim)
    xr.testing.assert_equal(result, expected)


@pytest.mark.parametrize(
    "valid_dataset",
    ["valid_poses_dataset", "valid_bboxes_dataset"],
)
@pytest.mark.parametrize(
    "selection_fn",
    [
        lambda position: (
            position.isel(individuals=0),
            position.isel(individuals=1),
        ),
        lambda position: (
            position.where(
                position.individuals == position.individuals[0], drop=True
            ).squeeze(),
            position.where(
                position.individuals == position.individuals[1], drop=True
            ).squeeze(),
        ),
        lambda position: (
            position.isel(individuals=0, keypoints=0),
            position.isel(individuals=1, keypoints=0),
        ),
        lambda position: (
            position.where(
                position.keypoints == position.keypoints[0], drop=True
            ).isel(individuals=0),
            position.where(
                position.keypoints == position.keypoints[0], drop=True
            ).isel(individuals=1),
        ),
    ],
    ids=[
        "dim_has_ndim_0",
        "dim_has_ndim_1",
        "labels_dim_has_ndim_0",
        "labels_dim_has_ndim_1",
    ],
)
def test_cdist_with_single_dim_inputs(valid_dataset, selection_fn, request):
    if request.node.callspec.id not in [
        "labels_dim_has_ndim_0-valid_bboxes_dataset",
        "labels_dim_has_ndim_1-valid_bboxes_dataset",
    ]:
        valid_dataset = request.getfixturevalue(valid_dataset)
        position = valid_dataset.position
        a, b = selection_fn(position)
        assert isinstance(_cdist(a, b, "individuals"), xr.DataArray)


@pytest.mark.parametrize(
    "dim, pairs, expected_data_vars",
    [
        ("individuals", {"id_0": ["id_1"]}, None),
        ("individuals", {"id_0": "id_1"}, None),
        (
            "individuals",
            {"id_0": ["id_1"], "id_1": "id_0"},
            [("id_0", "id_1"), ("id_1", "id_0")],
        ),
        ("individuals", "all", None),
        ("keypoints", {"centroid": ["left"]}, None),
        ("keypoints", {"centroid": "left"}, None),
        (
            "keypoints",
            {"centroid": ["left"], "left": "right"},
            [("centroid", "left"), ("left", "right")],
        ),
        (
            "keypoints",
            "all",
            [("centroid", "left"), ("centroid", "right"), ("left", "right")],
        ),
    ],
)
def test_compute_pairwise_distances_with_valid_pairs(
    valid_poses_dataset, dim, pairs, expected_data_vars
):
    result = compute_pairwise_distances(
        valid_poses_dataset.position, dim, pairs
    )
    if isinstance(result, dict):
        expected_data_vars = [
            f"dist_{pair[0]}_{pair[1]}" for pair in expected_data_vars
        ]
        assert set(result.keys()) == set(expected_data_vars)
    else:
        assert isinstance(result, xr.DataArray)


@pytest.mark.parametrize(
    "ds, dim, pairs",
    [
        ("valid_poses_dataset", "invalid_dim", {"id_0": "id_1"}),
        ("valid_poses_dataset", "keypoints", "invalid_string"),
        ("valid_poses_dataset", "individuals", {}),
        ("missing_dim_poses_dataset", "keypoints", "all"),
        ("missing_dim_bboxes_dataset", "individuals", "all"),
    ],
)
def test_compute_pairwise_distances_with_invalid_input(
    ds, dim, pairs, request
):
    with pytest.raises(ValueError):
        compute_pairwise_distances(
            request.getfixturevalue(ds).position, dim, pairs
        )
