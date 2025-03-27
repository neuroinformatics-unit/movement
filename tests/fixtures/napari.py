import numpy as np
import pytest

from movement.io import save_poses


@pytest.fixture
def valid_poses_path_and_ds_with_localised_nans(valid_poses_dataset, tmp_path):
    """Return a factory of (path, dataset) pairs representing
    valid pose datasets with NaN values at specific locations.
    """
    # Make a deep-copy of the valid dataset to avoid modifying the
    # original fixture
    ds = valid_poses_dataset.copy(deep=True)

    def _valid_poses_path_and_ds_with_localised_nans(
        nan_location, filename="ds_with_nans.csv"
    ):
        """Return a valid poses dataset and corresponding file with NaN values
        at specific locations.

        The ``nan_location`` parameter is a dictionary that specifies which
        coordinates to set to NaN.

        The dataset is modified from the `valid_poses_dataset` which represents
        2 individuals ("id_0" and "id_1") with up to 3 keypoints ("centroid",
        "left", "right") moving in uniform linear motion for 10 frames in 2D
        space.
        """
        # Express NaN location in time in "time" coordinates
        if nan_location["time"] == "start":
            time_point = 0
        elif nan_location["time"] == "middle":
            time_point = ds.coords["time"][ds.coords["time"].shape[0] // 2]
        elif nan_location["time"] == "end":
            time_point = ds.coords["time"][-1]

        # Set the selected values to NaN
        ds.position.loc[
            {
                "individuals": nan_location["individuals"],
                "keypoints": nan_location["keypoints"],
                "time": time_point,
            }
        ] = np.nan

        # Export as a DLC-csv file
        out_path = tmp_path / filename
        save_poses.to_dlc_file(ds, out_path, split_individuals=False)

        return (out_path, ds)

    return _valid_poses_path_and_ds_with_localised_nans


@pytest.fixture
def valid_poses_path_and_ds(valid_poses_dataset, tmp_path):
    """Return a (path, dataset) pair representing a poses dataset
    with 2 individuals ("id_0" and "id_1") and 3 keypoints
    ("centroid", "left", "right") moving in uniform linear
    motion for 10 frames in 2D space.
    """
    out_path = tmp_path / "ds.csv"
    save_poses.to_dlc_file(valid_poses_dataset, out_path)
    return (out_path, valid_poses_dataset)


@pytest.fixture
def valid_poses_path_and_ds_nan_start(
    valid_poses_path_and_ds_with_localised_nans,
):
    """Return a (path, dataset) pair representing a poses dataset
    with 2 individuals ("id_0" and "id_1") and 3 keypoints
    ("centroid", "left", "right") moving in uniform linear
    motion for 10 frames in 2D space, with all NaN values for the
    first frame.
    """
    out_path, ds = valid_poses_path_and_ds_with_localised_nans(
        {
            "time": "start",
            "individuals": ["id_0", "id_1"],
            "keypoints": ["centroid", "left", "right"],
        },
        filename="ds_with_nan_start.csv",
    )
    return (out_path, ds)


@pytest.fixture
def valid_poses_path_and_ds_nan_end(
    valid_poses_path_and_ds_with_localised_nans,
):
    """Return a (path, dataset) pair representing a poses dataset
    with 2 individuals ("id_0" and "id_1") and 3 keypoints
    ("centroid", "left", "right") moving in uniform linear
    motion for 10 frames in 2D space, with all NaN values for the
    last frame.
    """
    out_path, ds = valid_poses_path_and_ds_with_localised_nans(
        {
            "time": "end",
            "individuals": ["id_0", "id_1"],
            "keypoints": ["centroid", "left", "right"],
        },
        filename="ds_with_nan_end.csv",
    )
    return (out_path, ds)


@pytest.fixture
def valid_poses_path_and_ds_short(valid_poses_dataset, tmp_path):
    """Return a valid poses dataset with 2 individuals and 3 keypoints
    moving in uniform linear motion for 5 frames in 2D space.
    The dataset is modified from the `valid_poses_dataset` which represents
    2 individuals ("id_0" and "id_1") with up to 3 keypoints ("centroid",
    "left", "right") moving in uniform linear motion for 10 frames in 2D
    space.
    """
    # Modify the dataset to have only 5 frames
    valid_poses_dataset = valid_poses_dataset.sel(time=slice(0, 5))

    # Export as a DLC-csv file
    out_path = tmp_path / "ds_short.csv"
    save_poses.to_dlc_file(
        valid_poses_dataset, out_path, split_individuals=False
    )

    return (out_path, valid_poses_dataset)
