"""Valid and invalid data fixtures."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from movement.validators.datasets import ValidBboxesInputs, ValidPosesInputs


# -------------------- Valid bboxes datasets and arrays --------------------
@pytest.fixture
def valid_bboxes_arrays_all_zeros():
    """Return a dictionary of valid zero arrays (in terms of shape) for a
    valid bboxes dataset.
    """
    # define the shape of the arrays
    n_frames, n_space, n_individuals = (10, 2, 2)

    # build a valid array for position or shape with all zeros
    valid_bbox_array_all_zeros = np.zeros((n_frames, n_space, n_individuals))

    # return as a dict
    return {
        "position": valid_bbox_array_all_zeros,
        "shape": valid_bbox_array_all_zeros,
        "individual_names": ["id_" + str(id) for id in range(n_individuals)],
    }


@pytest.fixture
def valid_bboxes_arrays():
    """Return a dictionary of valid arrays for a
    ValidBboxesInputs representing a uniform linear motion.

    It represents 2 individuals for 10 frames, in 2D space.
    - Individual 0 moves along the x=y line from the origin.
    - Individual 1 moves along the x=-y line line from the origin.

    All confidence values are set to 0.9 except the following which are set
    to 0.1:
    - Individual 0 at frames 2, 3, 4
    - Individual 1 at frames 2, 3
    """
    # define the shape of the arrays
    n_frames, n_space, n_individuals = (10, 2, 2)

    # build a valid array for position
    # make bbox with id_i move along x=((-1)**(i))*y line from the origin
    # if i is even: along x = y line
    # if i is odd: along x = -y line
    # moving one unit along each axis in each frame
    position = np.zeros((n_frames, n_space, n_individuals))
    for i in range(n_individuals):
        position[:, 0, i] = np.arange(n_frames)
        position[:, 1, i] = (-1) ** i * np.arange(n_frames)

    # build a valid array for constant bbox shape (60, 40)
    constant_shape = float(60), float(40)  # width, height in pixels
    shape = np.tile(constant_shape, (n_frames, n_individuals, 1)).transpose(
        0, 2, 1
    )

    # build an array of confidence values, all 0.9
    confidence = np.full((n_frames, n_individuals), 0.9)

    # set 5 low-confidence values
    # - set 3 confidence values for bbox id_0 to 0.1
    # - set 2 confidence values for bbox id_1 to 0.1
    idx_start = 2
    confidence[idx_start : idx_start + 3, 0] = 0.1
    confidence[idx_start : idx_start + 2, 1] = 0.1

    return {
        "position": position,
        "shape": shape,
        "confidence": confidence,
    }


@pytest.fixture
def valid_bboxes_dataset(valid_bboxes_arrays):
    """Return a valid bboxes dataset for two individuals moving in uniform
    linear motion, with 5 frames with low confidence values and time in frames.

    It represents 2 individuals for 10 frames, in 2D space.
    - Individual 0 moves along the x=y line from the origin,
      in the x positive, y positive quadrant.
    - Individual 1 moves along the x=-y line line from the origin,
      in the x positive, y negative quadrant.

    All confidence values are set to 0.9 except the following which are set
    to 0.1:
    - Individual 0 at frames 2, 3, 4
    - Individual 1 at frames 2, 3
    """
    dim_names = ValidBboxesInputs.DIM_NAMES

    position_array = valid_bboxes_arrays["position"]
    shape_array = valid_bboxes_arrays["shape"]
    confidence_array = valid_bboxes_arrays["confidence"]

    n_frames, n_individuals, _ = position_array.shape

    return xr.Dataset(
        data_vars={
            "position": xr.DataArray(position_array, dims=dim_names),
            "shape": xr.DataArray(shape_array, dims=dim_names),
            "confidence": xr.DataArray(
                confidence_array, dims=dim_names[:1] + dim_names[2:]
            ),
        },
        coords={
            dim_names[0]: np.arange(n_frames),
            dim_names[1]: ["x", "y"],
            dim_names[2]: [f"id_{id}" for id in range(n_individuals)],
        },
        attrs={
            "time_unit": "frames",
            "source_software": "test",
            "source_file": "test_bboxes.csv",
            "ds_type": "bboxes",
        },
    )


@pytest.fixture
def valid_bboxes_dataset_in_seconds(valid_bboxes_dataset):
    """Return a valid bboxes dataset with time in seconds.

    The origin of time is assumed to be time = frame 0 = 0 seconds.
    The time unit is set to "seconds" and the fps is set to 60.
    """
    fps = 60
    valid_bboxes_dataset["time"] = valid_bboxes_dataset.time / fps
    valid_bboxes_dataset.attrs["time_unit"] = "seconds"
    valid_bboxes_dataset.attrs["fps"] = fps
    return valid_bboxes_dataset


@pytest.fixture
def valid_bboxes_dataset_with_nan(valid_bboxes_dataset):
    """Return a valid bboxes dataset with NaN values in the position array."""
    # Set 3 NaN values in the position array for id_0
    valid_bboxes_dataset.position.loc[
        {"individual": "id_0", "time": [3, 7, 8]}
    ] = np.nan
    return valid_bboxes_dataset


# -------------------- Valid poses datasets and arrays --------------------
@pytest.fixture
def valid_poses_arrays():
    """Return a dictionary of valid arrays for a
    valid poses dataset representing a uniform linear motion.

    This fixture is a factory of fixtures.
    Depending on the ``array_type`` requested (``multi_individual_array``,
    ``single_keypoint_array``, or ``single_individual_array``),
    the returned array can represent up to 2 individuals with
    up to 3 keypoints, moving at constant velocity for 10 frames in 2D space.
    Default is a ``multi_individual_array`` (2 individuals, 3 keypoints each).
    At each frame the individuals cover a distance of sqrt(2) in x-y space.
    Specifically:
    - Individual 0 moves along the x=y line from the origin.
    - Individual 1 moves along the x=-y line line from the origin.

    All confidence values for all keypoints are set to 0.9 except
    for the "centroid" (index=0) at the following frames,
    which are set to 0.1:
    - Individual 0 at frames 2, 3, 4
    - Individual 1 at frames 2, 3
    """

    def _valid_poses_arrays(array_type):
        """Return a dictionary of valid arrays for ValidPosesInputs."""
        # Unless specified, default is a ``multi_individual_array`` with
        # 10 frames, 3 keypoints, and 2 individuals in 2D space.
        n_frames, n_space, n_keypoints, n_individuals = (10, 2, 3, 2)

        # define centroid (index=0) trajectory in position array
        # for each individual, the centroid moves along
        # the x=+/-y line, starting from the origin.
        # - individual 0 moves along x = y line
        # - individual 1 moves along x = -y line (if applicable)
        # They move one unit along x and y axes in each frame
        frames = np.arange(n_frames)
        position = np.zeros((n_frames, n_space, n_keypoints, n_individuals))
        position[:, 0, 0, :] = frames[:, None]  # reshape to (n_frames, 1)
        position[:, 1, 0, 0] = frames
        position[:, 1, 0, 1] = -frames

        # define trajectory of left and right keypoints
        # for individual 0, at each timepoint:
        # - the left keypoint (index=1) is at x_centroid, y_centroid + 1
        # - the right keypoint (index=2) is at x_centroid + 1, y_centroid
        # for individual 1, at each timepoint:
        # - the left keypoint (index=1) is at x_centroid - 1, y_centroid
        # - the right keypoint (index=2) is at x_centroid, y_centroid + 1
        offsets = [
            [
                (0, 1),
                (1, 0),
            ],  # individual 0: left, right keypoints (x,y) offsets
            [
                (-1, 0),
                (0, 1),
            ],  # individual 1: left, right keypoints (x,y) offsets
        ]
        for i in range(n_individuals):
            for kpt in range(1, n_keypoints):
                position[:, 0, kpt, i] = (
                    position[:, 0, 0, i] + offsets[i][kpt - 1][0]
                )
                position[:, 1, kpt, i] = (
                    position[:, 1, 0, i] + offsets[i][kpt - 1][1]
                )

        # build an array of confidence values, all 0.9
        confidence = np.full((n_frames, n_keypoints, n_individuals), 0.9)
        # set 5 low-confidence values
        # - set 3 confidence values for individual id_0's centroid to 0.1
        # - set 2 confidence values for individual id_1's centroid to 0.1
        idx_start = 2
        confidence[idx_start : idx_start + 3, 0, 0] = 0.1
        confidence[idx_start : idx_start + 2, 0, 1] = 0.1

        if array_type == "single_keypoint_array":
            # return only the centroid keypoint
            position = position[:, :, :1, :]
            confidence = confidence[:, :1, :]
        elif array_type == "single_individual_array":
            # return only the first individual
            position = position[:, :, :, :1]
            confidence = confidence[:, :, :1]
        return {"position": position, "confidence": confidence}

    return _valid_poses_arrays


@pytest.fixture
def valid_poses_dataset(valid_poses_arrays, request):
    """Return a valid poses dataset.

    Depending on the ``array_type`` requested (``multi_individual_array``,
    ``single_keypoint_array``, or ``single_individual_array``),
    the dataset can represent up to 2 individuals ("id_0" and "id_1")
    with up to 3 keypoints ("centroid", "left", "right")
    moving in uniform linear motion for 10 frames in 2D space.
    Default is a ``multi_individual_array`` (2 individuals, 3 keypoints each).
    See the ``valid_poses_arrays`` fixture for details.
    """
    dim_names = ValidPosesInputs.DIM_NAMES
    # create a multi_individual_array by default unless overridden via param
    try:
        array_type = request.param
    except AttributeError:
        array_type = "multi_individual_array"
    poses_array = valid_poses_arrays(array_type)
    position_array = poses_array["position"]
    confidence_array = poses_array["confidence"]
    n_frames, _, n_keypoints, n_individuals = position_array.shape
    return xr.Dataset(
        data_vars={
            "position": xr.DataArray(position_array, dims=dim_names),
            "confidence": xr.DataArray(
                confidence_array, dims=dim_names[:1] + dim_names[2:]
            ),
        },
        coords={
            dim_names[0]: np.arange(n_frames),
            dim_names[1]: ["x", "y"],
            dim_names[2]: ["centroid", "left", "right"][:n_keypoints],
            dim_names[3]: [f"id_{i}" for i in range(n_individuals)],
        },
        attrs={
            "time_unit": "frames",
            "source_software": "test",
            "source_file": "test_poses.h5",
            "ds_type": "poses",
        },
    )


@pytest.fixture
def valid_poses_dataset_with_nan(valid_poses_dataset):
    """Return a valid poses dataset with NaNs introduced in the position array.

    Using ``valid_poses_dataset`` as the base dataset,
    the following NaN values are introduced:
    - Individual "id_0":
        - 3 NaNs in the centroid keypoint of individual id_0 (frames 3, 7, 8)
        - 1 NaN in the left keypoint of individual id_0 at time=0
        - 10 NaNs in the right keypoint of individual id_0 (all frames)
    - Individual "id_1" has no missing values.
    """
    valid_poses_dataset.position.loc[
        {"individual": "id_0", "keypoint": "centroid", "time": [3, 7, 8]}
    ] = np.nan
    valid_poses_dataset.position.loc[
        {"individual": "id_0", "keypoint": "left", "time": 0}
    ] = np.nan
    valid_poses_dataset.position.loc[
        {"individual": "id_0", "keypoint": "right"}
    ] = np.nan
    return valid_poses_dataset


@pytest.fixture
def valid_dlc_poses_df():
    """Return a valid DLC-style poses DataFrame."""
    return pd.read_hdf(pytest.DATA_PATHS.get("DLC_single-wasp.predictions.h5"))


@pytest.fixture
def valid_dlc_3d_poses_df(valid_dlc_poses_df):
    """Mock and return a valid DLC-style 3D poses DataFrame.

    The only difference between 2D and 3D DLC DataFrames is that
    the coordinate level in the columns MultiIndex includes 'z' instead of
    'likelihood'.
    """
    cols = [
        (scorer, bodypart, "z" if coord == "likelihood" else coord)
        for scorer, bodypart, coord in valid_dlc_poses_df.columns.to_list()
    ]
    valid_dlc_poses_df.columns = pd.MultiIndex.from_tuples(
        cols, names=valid_dlc_poses_df.columns.names
    )
    return valid_dlc_poses_df


# -------------------- Invalid bboxes datasets --------------------
@pytest.fixture
def missing_var_bboxes_dataset(valid_bboxes_dataset):
    """Return a bboxes dataset missing the required position variable."""
    return valid_bboxes_dataset.drop_vars("position")


@pytest.fixture
def missing_two_vars_bboxes_dataset(valid_bboxes_dataset):
    """Return a bboxes dataset missing the required position
    and shape variables.
    """
    return valid_bboxes_dataset.drop_vars(["position", "shape"])


@pytest.fixture
def missing_dim_bboxes_dataset(valid_bboxes_dataset):
    """Return a bboxes dataset missing the required time dimension."""
    return valid_bboxes_dataset.rename({"time": "tame"})


@pytest.fixture
def missing_two_dims_bboxes_dataset(valid_bboxes_dataset):
    """Return a bboxes dataset missing the required time
    and space dimensions.
    """
    return valid_bboxes_dataset.rename({"time": "tame", "space": "spice"})


# -------------------- Invalid poses datasets --------------------
@pytest.fixture
def not_a_dataset():
    """Return data that is not a pose tracks dataset."""
    return [1, 2, 3]


@pytest.fixture
def empty_dataset():
    """Return an empty pose tracks dataset."""
    return xr.Dataset()


@pytest.fixture
def missing_var_poses_dataset(valid_poses_dataset):
    """Return a poses dataset missing the required position variable."""
    return valid_poses_dataset.drop_vars("position")


@pytest.fixture
def missing_dim_poses_dataset(valid_poses_dataset):
    """Return a poses dataset missing the required time dimension."""
    return valid_poses_dataset.rename({"time": "tame"})
