import numpy as np
import pytest
import xarray as xr

from movement.analysis import navigation


@pytest.fixture
def mock_dataset():
    """Return a mock DataArray containing four known head orientations."""
    time = np.array([0, 1, 2, 3])
    individuals = np.array(["individual_0"])
    keypoints = np.array(["left_ear", "right_ear", "nose"])
    space = np.array(["x", "y"])

    ds = xr.DataArray(
        [
            [[[1, 0], [-1, 0], [0, -1]]],  # time 0
            [[[0, 1], [0, -1], [1, 0]]],  # time 1
            [[[-1, 0], [1, 0], [0, 1]]],  # time 2
            [[[0, -1], [0, 1], [-1, 0]]],  # time 3
        ],
        dims=["time", "individuals", "keypoints", "space"],
        coords={
            "time": time,
            "individuals": individuals,
            "keypoints": keypoints,
            "space": space,
        },
    )
    return ds


def test_compute_head_direction_vector(mock_dataset):
    """Test that the correct head direction vectors
    are computed from a basic mock dataset.
    """
    # Test that validators work
    with pytest.raises(TypeError):
        np_array = [
            [[[1, 0], [-1, 0], [0, -1]]],
            [[[0, 1], [0, -1], [1, 0]]],
            [[[-1, 0], [1, 0], [0, 1]]],
            [[[0, -1], [0, 1], [-1, 0]]],
        ]
        navigation.compute_head_direction_vector(
            np_array, "left_ear", "right_ear"
        )

    with pytest.raises(AttributeError):
        mock_dataset_keypoint = mock_dataset.sel(keypoints="nose", drop=True)
        navigation.compute_head_direction_vector(
            mock_dataset_keypoint, "left_ear", "right_ear"
        )

    with pytest.raises(ValueError):
        navigation.compute_head_direction_vector(
            mock_dataset, "left_ear", "left_ear"
        )

    # Test that output contains correct datatype, dimensions, and values
    head_vector = navigation.compute_head_direction_vector(
        mock_dataset, "left_ear", "right_ear"
    )
    known_vectors = np.array([[[0, 2]], [[-2, 0]], [[0, -2]], [[2, 0]]])

    assert (
        isinstance(head_vector, xr.DataArray)
        and ("space" in head_vector.dims)
        and ("keypoints" not in head_vector.dims)
    )
    assert np.equal(head_vector.values, known_vectors).all()
