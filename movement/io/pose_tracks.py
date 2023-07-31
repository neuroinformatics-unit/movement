import logging
from pathlib import Path
from typing import ClassVar, List, Optional, Union

import h5py
import numpy as np
import pandas as pd
import xarray as xr
from pydantic import ValidationError
from sleap_io.io.slp import read_labels

from movement.io.file_validators import ValidFile, ValidHDF5, ValidPosesCSV

# get logger
logger = logging.getLogger(__name__)


class PoseTracks(xr.Dataset):
    """Dataset containing pose tracks and point-wise confidence scores.

    This is an `xarray.Dataset` object, with the following dimensions:
    - `time`: the number of frames in the video
    - `individuals`: the number of individuals in the video
    - `keypoints`: the number of keypoints in the skeleton
    - `space`: the number of spatial dimensions, either 2 or 3

    Appropriate coordinate labels are assigned to each dimension:
    list of unique names (str) for `individuals` and `keypoints`,
    ['x','y',('z')] for `space`. The coordinates of the `time` dimension are
    in seconds if `fps` is provided, otherwise they are in frame numbers.

    The dataset contains two data variables (`xarray.DataArray` objects):
    - `pose_tracks`: with shape (`time`, `individuals`, `keypoints`, `space`)
    - `confidence_scores`: with shape (`time`, `individuals`, `keypoints`)

    The dataset may also contain following attributes as metadata:
    - `fps`: the number of frames per second in the video
    - `time_unit`: the unit of the `time` coordinates, frames or seconds
    - `source_software`: the software from which the pose tracks were loaded
    - `source_file`: the file from which the pose tracks were loaded
    """

    dim_names: ClassVar[tuple] = (
        "time",
        "individuals",
        "keypoints",
        "space",
    )

    __slots__ = ("fps", "time_unit", "source_software", "source_file")

    def __init__(
        self,
        tracks_array: np.ndarray,
        scores_array: Optional[np.ndarray] = None,
        individual_names: Optional[List[str]] = None,
        keypoint_names: Optional[List[str]] = None,
        fps: Optional[float] = None,
    ):
        """Create a `PoseTracks` dataset containing pose tracks and
        point-wise confidence scores.

        Parameters
        ----------
        tracks_array : np.ndarray
            Array of shape (n_frames, n_individuals, n_keypoints, n_space)
            containing the pose tracks. It will be converted to a
            `xarray.DataArray` object named "pose_tracks".
        scores_array : np.ndarray, optional
            Array of shape (n_frames, n_individuals, n_keypoints) containing
            the point-wise confidence scores. It will be converted to a
            `xarray.DataArray` object named "confidence_scores".
            If None (default), the scores will be set to an array of NaNs.
        individual_names : list of str, optional
            List of unique names for the individuals in the video. If None
            (default), the individuals will be named "individual_0",
            "individual_1", etc.
        keypoint_names : list of str, optional
            List of unique names for the keypoints in the skeleton. If None
            (default), the keypoints will be named "keypoint_0", "keypoint_1",
            etc.
        fps : float, optional
            The number of frames per second in the video. If None (default),
            the `time` coordinates will be in frame numbers.
        """

        n_frames, n_individuals, n_keypoints, n_space = tracks_array.shape
        if scores_array is None:
            scores_array = np.full(
                (n_frames, n_individuals, n_keypoints), np.nan, dtype="float32"
            )
        if individual_names is None:
            individual_names = [
                f"individual_{i}" for i in range(n_individuals)
            ]
        if keypoint_names is None:
            keypoint_names = [f"keypoint_{i}" for i in range(n_keypoints)]
        if (fps is not None) and (fps <= 0):
            logger.warning(
                f"Expected fps to be a positive number, but got {fps}. "
                "Setting fps to None."
            )
            fps = None

        # Convert the pose tracks and confidence scores to xarray.DataArray
        tracks_da = xr.DataArray(tracks_array, dims=self.dim_names)
        scores_da = xr.DataArray(scores_array, dims=self.dim_names[:-1])

        # Create the time coordinate, depending on the value of fps
        time_coords = np.arange(n_frames, dtype=int)
        time_unit = "frames"
        if fps is not None:
            time_coords = time_coords / fps
            time_unit = "seconds"

        # Combine the DataArrays into a Dataset, with common coordinates
        super().__init__(
            data_vars={
                "pose_tracks": tracks_da,
                "confidence_scores": scores_da,
            },
            coords={
                self.dim_names[0]: time_coords,
                self.dim_names[1]: individual_names,
                self.dim_names[2]: keypoint_names,
                self.dim_names[3]: ["x", "y", "z"][:n_space],
            },
            attrs={
                "fps": fps,
                "time_unit": time_unit,
                "source_software": None,
                "source_file": None,
            },
        )

    @classmethod
    def from_dlc_df(cls, df: pd.DataFrame, fps: Optional[float] = None):
        """Create a `PoseTracks` dataset from a DLC_style pandas DataFrame.

        Parameters
        ----------
        df : pandas DataFrame
            DataFrame containing the pose tracks and confidence scores. Must
            be formatted as in DeepLabCut output files (see Notes).
        fps : float, optional
            The number of frames per second in the video. If None (default),
            the `time` coordinates will be in frame numbers.

        Notes
        -----
        The DataFrame must have a multi-index column with the following levels:
        "scorer", ("individuals"), "bodyparts", "coords". The "individuals"
        level may be omitted if there is only one individual in the video.
        The "coords" level contains the spatial coordinates "x", "y",
        as well as "likelihood" (point-wise confidence scores).
        The row index corresponds to the frame number.
        """

        # read names of individuals and keypoints from the DataFrame
        if "individuals" in df.columns.names:
            individual_names = (
                df.columns.get_level_values("individuals").unique().to_list()
            )
        else:
            individual_names = ["individual_0"]

        keypoint_names = (
            df.columns.get_level_values("bodyparts").unique().to_list()
        )

        # reshape the data into (n_frames, n_individuals, n_keypoints, 3)
        # where the last axis contains "x", "y", "likelihood"
        tracks_with_scores = df.to_numpy().reshape(
            (-1, len(individual_names), len(keypoint_names), 3)
        )

        return cls(
            tracks_array=tracks_with_scores[:, :, :, :-1],
            scores_array=tracks_with_scores[:, :, :, -1],
            individual_names=individual_names,
            keypoint_names=keypoint_names,
            fps=fps,
        )

    @classmethod
    def from_sleap_file(
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
            the `time` coordinates will be in frame numbers.


        Notes
        -----
        The SLEAP predictions are normally saved in a ".slp" file, e.g.
        "v1.predictions.slp". If this file contains both user-labeled and
        predicted instances, only the predicted ones will be loaded.

        An analysis file, suffixed with ".h5" can be exported from the ".slp"
        file, using either the command line tool `sleap-convert` (with the
        "--format analysis" option enabled) or the SLEAP GUI (Choose
        "Export Analysis HDF5…" from the "File" menu) [1]_.

        `movement` expects the tracks to be proofread before loading them,
        meaning each track is interpreted as a single individual/animal.
        Follow the SLEAP guide for tracking and proofreading [2]_.

        References
        ----------
        .. [1] https://sleap.ai/tutorials/analysis.html
        .. [2] https://sleap.ai/guides/proofreading.html

        Examples
        --------
        >>> from movement.io import PoseTracks
        >>> poses = PoseTracks.from_sleap_file("path/to/file.slp", fps=30)
        """

        # Validate the file path
        try:
            file = ValidFile(
                path=file_path,
                expected_permission="r",
                expected_suffix=[".h5", ".slp"],
            )
        except ValidationError as error:
            logger.error(error)
            raise error

        # Load data into a dictionary
        if file.path.suffix == ".h5":
            data_dict = cls._load_dict_from_sleap_analysis_file(file)
        else:  # file.path.suffix == ".slp"
            data_dict = cls._load_dict_from_sleap_labels_file(file)

        logger.debug(
            f"Loaded pose tracks from {file.path.as_posix()} into a dict."
        )

        # Initialize a PoseTracks dataset from the dictionary
        ds = cls(**data_dict, fps=fps)

        # Add metadata as attrs
        ds.attrs["source_software"] = "SLEAP"
        ds.attrs["source_file"] = file.path.as_posix()

        logger.info(f"Loaded pose tracks from {file.path}:")
        logger.info(ds)
        return ds

    @classmethod
    def from_dlc_file(
        cls, file_path: Union[Path, str], fps: Optional[float] = None
    ):
        """Load pose tracking data from a DeepLabCut (DLC) output file.

        Parameters
        ----------
        file_path : pathlib Path or str
            Path to the file containing the DLC poses, either in ".h5"
            or ".csv" format.
        fps : float, optional
            The number of frames per second in the video. If None (default),
            the `time` coordinates will be in frame numbers.


        Examples
        --------
        >>> from movement.io import PoseTracks
        >>> poses = PoseTracks.from_dlc_file("path/to/file.h5", fps=30)
        """

        # Validate the file path
        try:
            file = ValidFile(
                path=file_path,
                expected_permission="r",
                expected_suffix=[".csv", ".h5"],
            )
        except ValidationError as error:
            logger.error(error)
            raise error

        # Load the DLC poses into a DataFrame
        if file.path.suffix == ".csv":
            df = cls._parse_dlc_csv_to_df(file)
        else:  # file.path.suffix == ".h5"
            df = cls._load_df_from_dlc_h5(file)

        logger.debug(
            f"Loaded poses from {file.path.as_posix()} into a DataFrame."
        )
        # Convert the DataFrame to a PoseTracks dataset
        ds = cls.from_dlc_df(df=df, fps=fps)

        # Add metadata as attrs
        ds.attrs["source_software"] = "DeepLabCut"
        ds.attrs["source_file"] = file.path.as_posix()

        logger.info(f"Loaded pose tracks from {file_path}:")
        logger.info(ds)
        return ds

    def to_dlc_df(self) -> pd.DataFrame:
        """Convert the PoseTracks dataset to a DeepLabCut-style pandas
        DataFrame with multi-index columns.
        See the Notes section of the `from_dlc_df()` method for details.

        Returns
        -------
        pandas DataFrame

        Notes
        -----
        The DataFrame will have a multi-index column with the following levels:
        "scorer", "individuals", "bodyparts", "coords" (even if there is only
        one individual present). Regardless of the provenance of the
        points-wise confidence scores, they will be referred to as
        "likelihood", and stored in the "coords" level (as DeepLabCut expects).
        """

        # Concatenate the pose tracks and confidence scores into one array
        tracks_with_scores = np.concatenate(
            (
                self.pose_tracks.data,
                self.confidence_scores.data[..., np.newaxis],
            ),
            axis=-1,
        )

        # Create the DLC-style multi-index columns
        # Use the DLC terminology: scorer, individuals, bodyparts, coords
        scorer = ["movement"]
        individuals = self.coords["individuals"].data.tolist()
        bodyparts = self.coords["keypoints"].data.tolist()
        # The confidence scores in DLC are referred to as "likelihood"
        coords = self.coords["space"].data.tolist() + ["likelihood"]

        index_levels = ["scorer", "individuals", "bodyparts", "coords"]
        columns = pd.MultiIndex.from_product(
            [scorer, individuals, bodyparts, coords], names=index_levels
        )
        df = pd.DataFrame(
            data=tracks_with_scores.reshape(self.dims["time"], -1),
            index=np.arange(self.dims["time"], dtype=int),
            columns=columns,
            dtype=float,
        )
        logger.info("Converted PoseTracks dataset to DLC-style DataFrame.")
        return df

    def to_dlc_file(self, file_path: Union[str, Path]):
        """Save the dataset to a DeepLabCut-style .h5 or .csv file

        Parameters
        ----------
        file_path : pathlib Path or str
            Path to the file to save the DLC poses to. The file extension
            must be either ".h5" (recommended) or ".csv".
        """

        # Validate the file path
        try:
            file = ValidFile(
                path=file_path,
                expected_permission="w",
                expected_suffix=[".csv", ".h5"],
            )
        except ValidationError as error:
            logger.error(error)
            raise error

        # Convert the PoseTracks dataset to a DataFrame
        df = self.to_dlc_df()
        if file.path.suffix == ".csv":
            df.to_csv(file.path, sep=",")
        else:  # file.path.suffix == ".h5"
            df.to_hdf(file.path, key="df_with_missing")
        logger.info(f"Saved PoseTracks dataset to {file.path.as_posix()}.")

    @staticmethod
    def _load_dict_from_sleap_analysis_file(file: ValidFile):
        """Load pose tracks and confidence scores from a SLEAP analysis
        file into a dictionary."""

        # Validate the hdf5 file
        try:
            ValidHDF5(file=file, expected_datasets=["tracks"])
        except ValidationError as error:
            logger.error(error)
            raise error

        with h5py.File(file.path, "r") as f:
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
                "tracks_array": tracks,
                "scores_array": scores,
                "individual_names": [n.decode() for n in f["track_names"][:]],
                "keypoint_names": [n.decode() for n in f["node_names"][:]],
            }

    @staticmethod
    def _load_dict_from_sleap_labels_file(file: ValidFile):
        """Load pose tracks and confidence scores from a SLEAP labels file
        into a dictionary."""

        # Validate the .slp file as an HDF5 file
        try:
            ValidHDF5(file=file, expected_datasets=["pred_points", "metadata"])
        except ValidationError as error:
            logger.error(error)
            raise error

        labels = read_labels(file.path.as_posix())
        tracks_with_scores = labels.numpy(return_confidence=True)

        return {
            "tracks_array": tracks_with_scores[:, :, :, :-1],
            "scores_array": tracks_with_scores[:, :, :, -1],
            "individual_names": [track.name for track in labels.tracks],
            "keypoint_names": [kp.name for kp in labels.skeletons[0].nodes],
        }

    @staticmethod
    def _parse_dlc_csv_to_df(file: ValidFile) -> pd.DataFrame:
        """If poses are loaded from a DeepLabCut.csv file, the DataFrame
        lacks the multi-index columns that are present in the .h5 file. This
        function parses the csv file to a pandas DataFrame with multi-index
        columns, i.e. the same format as in the .h5 file.
        """

        try:
            ValidPosesCSV(file=file, multianimal=False)
        except ValidationError as error:
            logger.error(error)
            raise error

        possible_level_names = ["scorer", "individuals", "bodyparts", "coords"]
        with open(file.path, "r") as f:
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
            file.path,
            skiprows=len(header_lines),
            index_col=0,
            names=np.array(columns),
        )
        df.columns.rename(level_names, inplace=True)
        return df

    @staticmethod
    def _load_df_from_dlc_h5(file: ValidFile) -> pd.DataFrame:
        """Load pose tracks and likelihood scores from a DeepLabCut .h5 file
        into a pandas DataFrame."""

        try:
            ValidHDF5(file=file, expected_datasets=["df_with_missing"])
        except ValidationError as error:
            logger.error(error)
            raise error

        try:
            # pd.read_hdf does not always return a DataFrame
            df = pd.DataFrame(pd.read_hdf(file.path, key="df_with_missing"))
        except Exception as error:
            logger.error(error)
            raise error

        return df
