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
    """Test the computation of pairwise distances with known values."""
    labels_dim = "keypoints" if dim == "individuals" else "individuals"
    input_dataarray = valid_poses_dataset.position.sel(
        time=slice(0, 1)
    )  # Use only the first two frames for simplicity
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
    assert result.name == "distance"
    xr.testing.assert_equal(
        result,
        expected,
    )


@pytest.mark.parametrize(
    "valid_dataset",
    ["valid_poses_dataset", "valid_bboxes_dataset"],
)
@pytest.mark.parametrize(
    "selection_fn",
    [
        # individuals dim is scalar,
        # poses: multiple keypoints
        # bboxes: missing keypoints dim
        # e.g. comparing 2 individuals from the same data array
        lambda position: (
            position.isel(individuals=0),
            position.isel(individuals=1),
        ),
        # individuals dim is 1D
        # poses: multiple keypoints
        # bboxes: missing keypoints dim
        # e.g. comparing 2 single-individual data arrays
        lambda position: (
            position.where(
                position.individuals == position.individuals[0], drop=True
            ).squeeze(),
            position.where(
                position.individuals == position.individuals[1], drop=True
            ).squeeze(),
        ),
        # both individuals and keypoints dims are scalar (poses only)
        # e.g. comparing 2 individuals from the same data array,
        # at the same keypoint
        lambda position: (
            position.isel(individuals=0, keypoints=0),
            position.isel(individuals=1, keypoints=0),
        ),
        # individuals dim is scalar, keypoints dim is 1D (poses only)
        # e.g. comparing 2 single-individual, single-keypoint data arrays
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
    """Test that the pairwise distances data array is successfully
     returned regardless of whether the input DataArrays have
    ``dim`` ("individuals") and ``labels_dim`` ("keypoints")
    being either scalar (ndim=0) or 1D (ndim=1),
    or if ``labels_dim`` is missing.
    """
    if request.node.callspec.id not in [
        "labels_dim_has_ndim_0-valid_bboxes_dataset",
        "labels_dim_has_ndim_1-valid_bboxes_dataset",
    ]:  # Skip tests with keypoints dim for bboxes
        valid_dataset = request.getfixturevalue(valid_dataset)
        position = valid_dataset.position
        a, b = selection_fn(position)
        result = _cdist(a, b, "individuals")
        assert result.name == "distance"
        assert isinstance(result, xr.DataArray)


@pytest.mark.parametrize(
    "dim, pairs, expected_data_vars",
    [
        ("individuals", {"id_0": ["id_1"]}, None),  # list input
        ("individuals", {"id_0": "id_1"}, None),  # string input
        (
            "individuals",
            {"id_0": ["id_1"], "id_1": "id_0"},
            [("id_0", "id_1"), ("id_1", "id_0")],
        ),
        ("individuals", "all", None),  # all pairs
        ("keypoints", {"centroid": ["left"]}, None),  # list input
        ("keypoints", {"centroid": "left"}, None),  # string input
        (
            "keypoints",
            {"centroid": ["left"], "left": "right"},
            [("centroid", "left"), ("left", "right")],
        ),
        (
            "keypoints",
            "all",
            [("centroid", "left"), ("centroid", "right"), ("left", "right")],
        ),  # all pairs
    ],
)
def test_compute_pairwise_distances_with_valid_pairs(
    valid_poses_dataset, dim, pairs, expected_data_vars
):
    """Test that the expected pairwise distances are computed
    for valid ``pairs`` inputs.
    """
    result = compute_pairwise_distances(
        valid_poses_dataset.position, dim, pairs
    )
    if isinstance(result, dict):
        for _, value in result.items():
            assert isinstance(value, xr.DataArray)
            assert value.name == "distance"
        expected_data_vars = [
            f"dist_{pair[0]}_{pair[1]}" for pair in expected_data_vars
        ]
        assert set(result.keys()) == set(expected_data_vars)
    else:  # expect single DataArray
        assert isinstance(result, xr.DataArray)
        assert result.name == "distance"


@pytest.mark.parametrize(
    "ds, dim, pairs",
    [
        (
            "valid_poses_dataset",
            "invalid_dim",
            {"id_0": "id_1"},
        ),  # invalid dim
        (
            "valid_poses_dataset",
            "keypoints",
            "invalid_string",
        ),  # invalid pairs
        ("valid_poses_dataset", "individuals", {}),  # empty pairs
        ("missing_dim_poses_dataset", "keypoints", "all"),  # invalid dataset
        (
            "missing_dim_bboxes_dataset",
            "individuals",
            "all",
        ),  # invalid dataset
    ],
)
def test_compute_pairwise_distances_with_invalid_input(
    ds, dim, pairs, request
):
    """Test that an error is raised for invalid inputs."""
    with pytest.raises(ValueError):
        compute_pairwise_distances(
            request.getfixturevalue(ds).position, dim, pairs
        )
