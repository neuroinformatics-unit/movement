import logging
from pathlib import Path
from typing import ClassVar, Optional, Union

import h5py
import numpy as np
import pandas as pd
import xarray as xr
from sleap_io.io.slp import read_labels

# get logger
logger = logging.getLogger(__name__)


class PoseTracks(xr.Dataset):
    """Dataset containing pose tracks and point-wise confidence scores.

    This is a `xarray.Dataset` object, with the following dimensions:
    - `frames`: the number of frames in the video
    - `individuals`: the number of individuals in the video
    - `keypoints`: the number of keypoints in the skeleton
    - `space`: the number of spatial dimensions, either 2 or 3

    Each dimension is assigned appropriate coordinates:
     - frame indices (int) for `frames`
     - list of unique names (str) for `individuals` and `keypoints`
     - `x`, `y` (and `z`) for `space`

    If `fps` is supplied, the `frames` dimension is also assigned a `time`
    coordinate. If `fps` is None, the temporal dimension can only be
    accessed through frame indices.

    The dataset contains two data variables (`xarray.DataArray` objects):
    - `pose_tracks`: with shape (`frames`, `individuals`, `keypoints`, `space`)
    - `confidence_scores`: with shape (`frames`, `individuals`, `keypoints`)

    The dataset may also contain following attributes as metadata:
    - `fps`: the number of frames per second in the video
    - `source_software`: the software from which the pose tracks were loaded
    - `source_file`: the file from which the pose tracks were loaded
    """

    dim_names: ClassVar[tuple] = (
        "frames",
        "individuals",
        "keypoints",
        "space",
    )

    __slots__ = ("fps", "source_software", "source_file")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def from_dict(
        cls,
        dict_: dict,
    ):
        """Create a `PosteTracks` dataset from a dictionary of pose tracks,
        confidence scores, and metadata.

        Parameters
        ----------
        dict_ : dict
            A dictionary with the following keys:
            - "pose_tracks": np.ndarray of shape (n_frames, n_individuals,
                n_keypoints, n_space_dims)
            - "confidence_scores": np.ndarray of shape (n_frames,
                n_individuals, n_keypoints)
            - "individual_names": list of strings, with length individuals
            - "keypoint_names": list of strings, with length n_keypoints
            - "fps": float, the number of frames per second in the video.
                If None, the "time" coordinate will not be added.
        """

        # Convert the pose tracks and confidence scores to xarray.DataArray
        tracks_da = xr.DataArray(dict_["pose_tracks"], dims=cls.dim_names)
        scores_da = xr.DataArray(
            dict_["confidence_scores"], dims=cls.dim_names[:-1]
        )

        # Combine the DataArrays into a Dataset, with common coordinates
        ds = cls(
            data_vars={
                "pose_tracks": tracks_da,
                "confidence_scores": scores_da,
            },
            coords={
                cls.dim_names[0]: np.arange(
                    dict_["pose_tracks"].shape[0], dtype=int
                ),
                cls.dim_names[1]: dict_["individual_names"],
                cls.dim_names[2]: dict_["keypoint_names"],
                cls.dim_names[3]: ["x", "y", "z"][
                    : dict_["pose_tracks"].shape[-1]
                ],
            },
            attrs={"fps": dict_["fps"]},
        )

        # If fps is given, create "time" coords for 1st ("frames") dimension
        if dict_["fps"] is not None:
            times = pd.TimedeltaIndex(
                ds.coords["frames"] / dict_["fps"], unit="s"
            )
            ds.coords["time"] = (cls.dim_names[0], times)

        return ds

    @classmethod
    def from_sleap(
        cls, file_path: Union[Path, str], fps: Optional[float] = None
    ):
        """Load pose tracking data from a SLEAP labels or analysis file.

        Parameters
        ----------
        file_path : pathlib Path or str
            Path to the file containing the SLEAP predictions, either in ".slp"
            or ".h5" (analysis) format. See Notes for more information.
        fps : float, optional
            The number of frames per second in the video. If None (default),
            the `time` coordinate will not be created.

        Notes
        -----
        The SLEAP inference procedure normally produces a file suffixed with
        ".slp" containing the predictions, e.g. "myproject.predictions.slp".
        This can be converted to an ".h5" (analysis) file using the command
        line tool `sleap-convert` with the "--format analysis" option enabled,
        or alternatively by choosing "Export Analysis HDF5…" from the "File"
        menu of the SLEAP GUI [1]_.

        If the ".slp" file contains both user-labeled and predicted instances,
        this function will only load the ones predicted by the SLEAP model

        `movement` expects the tracks to be proofread before loading them.
        There should be as many tracks as there are instances (animals) in the
        video, without identity switches. Follow the SLEAP guide for
        tracking and proofreading [2]_.

        References
        ----------
        .. [1] https://sleap.ai/tutorials/analysis.html
        .. [2] https://sleap.ai/guides/proofreading.html

        Examples
        --------
        >>> from movement.io import PoseTracks
        >>> poses = PoseTracks.from_sleap("path/to/v1.predictions.slp", fps=30)
        """

        if not isinstance(file_path, Path):
            file_path = Path(file_path)

        # Load data into a dictionary
        if file_path.suffix == ".h5":
            dict_ = cls._load_dict_from_sleap_analysis_file(file_path)
        elif file_path.suffix == ".slp":
            dict_ = cls._load_dict_from_sleap_labels_file(file_path)
        else:
            error_msg = (
                f"Expected file suffix to be '.h5' or '.slp', "
                f"but got '{file_path.suffix}'. Make sure the file is "
                "a SLEAP labels file with suffix '.slp' or SLEAP analysis "
                "file with suffix '.h5'."
            )
            # logger.error(error_msg)
            raise ValueError(error_msg)

        logger.debug(
            f"Loaded pose tracks from {file_path.as_posix()} into a dict."
        )

        # Initialize a PoseTracks dataset from the dictionary
        ds = cls.from_dict({**dict_, "fps": fps})

        # Add metadata as attrs
        ds.attrs["source_software"] = "SLEAP"
        ds.attrs["source_file"] = file_path.as_posix()
        return ds

    @staticmethod
    def _load_dict_from_sleap_analysis_file(file_path: Path):
        """Load pose tracks and confidence scores from a SLEAP analysis
        file into a dictionary."""

        with h5py.File(file_path, "r") as f:
            tracks = f["tracks"][:].T
            n_frames, n_keypoints, n_space, n_tracks = tracks.shape
            tracks = tracks.reshape((n_frames, n_tracks, n_keypoints, n_space))
            # Create an array of NaNs for the confidence scores
            scores = np.full(
                (n_frames, n_tracks, n_keypoints), np.nan, dtype="float32"
            )
            # If present, read the point-wise scores, and reshape them
            if "point_scores" in f.keys():
                scores = f["point_scores"][:].reshape(
                    (n_frames, n_tracks, n_keypoints)
                )

            return {
                "pose_tracks": tracks,
                "confidence_scores": scores,
                "individual_names": [n.decode() for n in f["track_names"][:]],
                "keypoint_names": [n.decode() for n in f["node_names"][:]],
            }

    @staticmethod
    def _load_dict_from_sleap_labels_file(file_path: Path):
        """Load pose tracks and confidence scores from a SLEAP labels file
        into a dictionary."""

        labels = read_labels(file_path.as_posix())
        tracks_with_scores = labels.numpy(return_confidence=True)

        return {
            "pose_tracks": tracks_with_scores[:, :, :, :-1],
            "confidence_scores": tracks_with_scores[:, :, :, -1],
            "individual_names": [track.name for track in labels.tracks],
            "keypoint_names": [kp.name for kp in labels.skeletons[0].nodes],
        }
