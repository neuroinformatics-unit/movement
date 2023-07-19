import logging
from pathlib import Path
from typing import ClassVar, Optional, Union

import h5py
import numpy as np
import pandas as pd
import xarray as xr
from sleap_io.io.slp import read_labels

from movement.io.validators import DeepLabCutPosesFile

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
    def from_dataframe(cls, df: pd.DataFrame, fps: Optional[float] = None):
        """Create a `PoseTracks` dataset from a pandas DataFrame.

        Parameters
        ----------
        df : pandas DataFrame
            DataFrame containing the pose tracks and confidence scores. Must
            be formatted as in DeepLabCut output files (see Notes).
        fps : float, optional
            The number of frames per second in the video. If None (default),
            the `time` coordinate will not be created.

        Notes
        -----
        The DataFrame must have a multi-index column with the following levels:
        "scorer", ("individuals"), "bodyparts", "coords".
        The "individuals level may be omitted if there is only one individual
        in the video. The "coords" level contains the spatial coordinates "x",
        "y", as well as "likelihood" (point-wise confidence scores). The row
        index corresponds to the frame number.

        Examples
        --------
        >>> from movement.io import PoseTracks
        >>> df = pd.read_csv("path/to/poses.csv")
        >>> poses = PoseTracks.from_dataframe(df, fps=30)
        """

        # Convert the DataFrame to a dictionary
        dict_ = cls.dataframe_to_dict(df)

        # Initialize a PoseTracks dataset from the dictionary
        ds = cls.from_dict({**dict_, "fps": fps})
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

    @classmethod
    def from_dlc(
        cls, file_path: Union[Path, str], fps: Optional[float] = None
    ):
        """Load pose tracking data from a DeepLabCut (DLC) output file.

        Parameters
        ----------
        file_path : pathlib Path or str
            Path to the file containing the DLC poses, either in ".slp"
            or ".h5" (analysis) format.
        fps : float, optional
            The number of frames per second in the video. If None (default),
            the `time` coordinate will not be created.


        Examples
        --------
        >>> from movement.io import PoseTracks
        >>> poses = PoseTracks.from_dlc("path/to/video_model.h5", fps=30)
        """

        # Validate the input file path
        dlc_poses_file = DeepLabCutPosesFile(file_path=file_path)
        file_suffix = dlc_poses_file.file_path.suffix

        # Load the DLC poses into a DataFrame
        try:
            if file_suffix == ".csv":
                df = cls._parse_dlc_csv_to_dataframe(dlc_poses_file.file_path)
            else:  # file can only be .h5 at this point
                df = pd.read_hdf(dlc_poses_file.file_path)
                # above line does not necessarily return a DataFrame
                df = pd.DataFrame(df)
        except (OSError, TypeError, ValueError) as e:
            error_msg = (
                f"Could not load poses from {file_path}. "
                "Please check that the file is valid and readable."
            )
            logger.error(error_msg)
            raise OSError from e
        logger.info(f"Loaded poses from {file_path} into a DataFrame.")

        # Convert the DataFrame to a PoseTracks dataset
        ds = cls.from_dataframe(df=df, fps=fps)

        # Add metadata as attrs
        ds.attrs["source_software"] = "DeepLabCut"
        ds.attrs["source_file"] = dlc_poses_file.filepath.as_posix()
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

    def _parse_dlc_csv_to_dataframe(file_path: Path) -> pd.DataFrame:
        """If poses are loaded from a DeepLabCut.csv file, the DataFrame
        lacks the multi-index columns that are present in the .h5 file. This
        function parses the csv file to a DataFrame with multi-index columns,
        i.e. the same format as in the .h5 file.

        Parameters
        ----------
        file_path : pathlib Path
            Path to the file containing the DLC poses, in .csv format.

        Returns
        -------
        pandas DataFrame
            DataFrame containing the DLC poses, with multi-index columns.
        """

        possible_level_names = ["scorer", "individuals", "bodyparts", "coords"]
        with open(file_path, "r") as f:
            # if line starts with a possible level name, split it into a list
            # of strings, and add it to the list of header lines
            header_lines = [
                line.strip().split(",")
                for line in f.readlines()
                if line.split(",")[0] in possible_level_names
            ]

        # Form multi-index column names from the header lines
        level_names = [line[0] for line in header_lines]
        column_tuples = list(zip(*[line[1:] for line in header_lines]))
        columns = pd.MultiIndex.from_tuples(column_tuples, names=level_names)

        # Import the DLC poses as a DataFrame
        df = pd.read_csv(
            file_path, skiprows=len(header_lines), index_col=0, names=columns
        )
        df.columns.rename(level_names, inplace=True)
        return df

    @staticmethod
    def dataframe_to_dict(df: pd.DataFrame) -> dict:
        """Convert a DeepLabCut-style DataFrame containing pose tracks and
        likelihood scores into a dictionary.

        Parameters
        ----------
        df : pandas DataFrame
            DataFrame formatted as in DeepLabCut output files.

        Returns
        -------
        dict
            Dictionary containing the pose tracks, confidence scores, and
            metadata.
        """

        # read names of individuals and keypoints from the DataFrame
        # retain the order of their appearance in the DataFrame
        if "individuals" in df.columns.names:
            ind_names = (
                df.columns.get_level_values("individuals").unique().to_list()
            )
        else:
            ind_names = ["individual_0"]

        kp_names = df.columns.get_level_values("bodyparts").unique().to_list()
        print(ind_names)
        print(kp_names)

        # reshape the data into (n_frames, n_individuals, n_keypoints, 3)
        # where the last axis contains "x", "y", "likelihood"
        tracks_with_scores = df.to_numpy().reshape(
            (-1, len(ind_names), len(kp_names), 3)
        )

        return {
            "pose_tracks": tracks_with_scores[:, :, :, :-1],
            "confidence_scores": tracks_with_scores[:, :, :, -1],
            "individual_names": ind_names,
            "keypoint_names": kp_names,
        }
