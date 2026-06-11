import string

import numpy as np
import pandas as pd
import pytest

from movement.io import save_poses


@pytest.fixture
def valid_poses_path_and_ds(valid_poses_dataset, tmp_path):
    """Return a (path, dataset) pair representing a poses dataset
    with data for 10 frames.

    The fixture is derived from the `valid_poses_dataset` fixture.
    """
    out_path = tmp_path / "ds.csv"
    save_poses.to_dlc_file(valid_poses_dataset, out_path)
    return (out_path, valid_poses_dataset)


@pytest.fixture
def valid_poses_path_and_ds_short(valid_poses_dataset, tmp_path):
    """Return a (path, dataset) pair representing a poses dataset
    with data for 5 frames.

    The fixture is derived from the `valid_poses_dataset` fixture.
    """
    # Modify the dataset to have only 5 frames
    valid_poses_dataset = valid_poses_dataset.sel(time=slice(0, 5))

    # Export as a DLC-csv file
    out_path = tmp_path / "ds_short.csv"
    save_poses.to_dlc_file(
        valid_poses_dataset, out_path, split_individuals=False
    )

    return (out_path, valid_poses_dataset)


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
                "individual": nan_location["individual"],
                "keypoint": nan_location["keypoint"],
                "time": time_point,
            }
        ] = np.nan

        # Export as a DLC-csv file
        out_path = tmp_path / filename
        save_poses.to_dlc_file(ds, out_path, split_individuals=False)

        return (out_path, ds)

    return _valid_poses_path_and_ds_with_localised_nans


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
            "individual": ["id_0", "id_1"],
            "keypoint": ["centroid", "left", "right"],
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
            "individual": ["id_0", "id_1"],
            "keypoint": ["centroid", "left", "right"],
        },
        filename="ds_with_nan_end.csv",
    )
    return (out_path, ds)


@pytest.fixture
def sample_layer_data(rng):
    """Return a dictionary of sample data for each napari layer type."""
    n_frames = 2000
    sample_points_data = rng.random((n_frames, 3))
    sample_image_data = rng.random((n_frames, 200, 200))
    sample_tracks_data = np.hstack(
        (
            np.tile([1, 2, 3, 4], (1, n_frames // 4)).T,
            rng.random((n_frames, 3)),
        )
    )
    sample_labels_data = rng.integers(0, 2, (200, 200))
    sample_shapes_data = rng.random((n_frames, 4, 2))  # rectangles
    sample_surface_data = (
        rng.random((4, 2)),  # vertices
        np.array([[0, 1, 2], [1, 2, 3]]),  # faces
        np.linspace(0, 1, 4),  # values
    )
    sample_vector_data = rng.random((100, 2, 2))

    return {
        "Points": sample_points_data,
        "Image": sample_image_data,
        "Tracks": sample_tracks_data,
        "Labels": sample_labels_data,
        "Shapes": sample_shapes_data,
        "Surface": sample_surface_data,
        "Vectors": sample_vector_data,
        "n_frames": n_frames,
    }


@pytest.fixture
def sample_properties_with_factorized():
    """Return a factory of properties dataframes with the specified property
    column and a factorized version of the same property (with the suffix
    "_factorized"). The maximum number of unique values for the property is 26.
    """

    def _sample_properties_with_factorized(property_name, n_unique_values):
        """Return a properties dataframe with the specified property column,
        some string values assigned to it and a factorized version of the same
        property (with the suffix "_factorized"). The maximum number of unique
        values is 26.
        """
        # Each unique value is repeated 3 times in the dataframe
        n_repeats = 3

        list_unique_properties = list(string.ascii_uppercase[:n_unique_values])
        properties_df = pd.DataFrame(
            {property_name: list_unique_properties * n_repeats}
        )

        # Factorize the color property
        codes, _ = pd.factorize(properties_df[property_name])
        properties_df[property_name + "_factorized"] = codes

        return properties_df

    return _sample_properties_with_factorized


# -------------------- Valid bboxes napari layers --------------------

# -------------------- Valid poses napari layers --------------------


@pytest.fixture
def valid_poses_napari_layers():
    """Return valid pose napari layers.

    Depending on the ``array_type`` the returned napari layer can represent:
    - ``multiple_individuals``: 2 individuals, 3 keypoints
    - ``single_individual``: 1 individual, 3 keypoints

    The simulated trajectories follow simple linear motion: individual 0 moves
    along ``x = y`` and individual 1 moves along ``x = y ``.

    Three keypoints are defined for each individual: ``centroid``, ``right`` and 
    ``left``. The ``left`` keypoint is offset by +1 in ``y`` relative to the 
    centroid. While the ``right`` keypoint is offset by +1 in ``x``. 

    The napari tracks array has shape ``(N, 4)`` with columns:
    ``track_id, frame, y, x``.
    """

    def _valid_poses_napari_layers(array_type):
        n_frames, n_keypoints, n_individuals = (10, 3, 2)
        keypoint_names = ["centroid", "left", "right"]

        if array_type == "single_individual":
            n_individuals = 1

        frames = np.arange(n_frames)

        napari_layers = []
        properties = []

        for individual_idx in range(n_individuals):
            for keypoint_idx, keypoint in enumerate(keypoint_names):
                for frame in frames:
                    # centroid trajectory
                    x = frame

                    if individual_idx == 0:
                        y = frame
                    else:
                        y = -frame

                    # keypoint offsets relative to centroid
                    if keypoint_idx == 1:  # left
                        y += 1
                    elif keypoint_idx == 2:  # right
                        x += 1

                    confidence = 0.9

                    # low-confidence centroid frames
                    if (
                        keypoint_idx == 0
                        and individual_idx == 0
                        and frame in [2, 3, 4]
                    ) or (
                        keypoint_idx == 0
                        and individual_idx == 1
                        and frame in [2, 3]
                    ):
                        confidence = 0.1

                    napari_layers.append(
                        [
                            individual_idx,  # track_id
                            frame,  # frame
                            y,  # y
                            x,  # x
                        ]
                    )

                    properties.append(
                        {
                            "individual": f"id_{individual_idx}",
                            "keypoint": keypoint,
                            "time": frame,
                            "confidence": confidence,
                        }
                    )

        return (
            np.asarray(napari_layers, dtype=np.float32),
            pd.DataFrame(properties),
        )

    return _valid_poses_napari_layers
