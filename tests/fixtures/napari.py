import numpy as np
import pytest

from movement.io import save_poses


@pytest.fixture
def valid_dataset_with_single_nan(valid_poses_dataset, tmp_path):
    """Return a factory of (path, dataset) pairs representing
    valid datasets with NaN values in the first, last or middle frame.
    """

    def _valid_dataset_with_single_nan(nan_location):
        """Return a valid dataset with a NaN position for a single
        individual in the first, last or middle frame.

        The `valid_poses_dataset` represents 2 individuals ("id_0" and "id_1")
        with up to 3 keypoints ("centroid", "left", "right")
        moving in uniform linear motion for 10 frames in 2D space.

        The `valid_bboxes_dataset` fixture represents 2 individuals
        ("id_0" and "id_1") moving in uniform linear motion, with 5
        frames with low confidence values and time in frames.
        """
        if nan_location["time"] == "start":
            time_point = 0
        elif nan_location["time"] == "middle":
            time_point = 2
        elif nan_location["time"] == "end":
            time_point = valid_poses_dataset.coords["time"][-1]

        # Set the selected position of id_0 to nan
        valid_poses_dataset.position.loc[
            {
                "individuals": nan_location["individuals"],
                "keypoints": nan_location["keypoints"],
                "time": time_point,
            }
        ] = np.nan

        # Export as SLEAP file (poses only)
        out_path = tmp_path / "sample_with_nan.csv"
        save_poses.to_dlc_file(
            valid_poses_dataset, out_path, split_individuals=False
        )

        return (out_path, valid_poses_dataset)

    return _valid_dataset_with_single_nan
