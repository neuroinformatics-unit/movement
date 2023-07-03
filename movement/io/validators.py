import logging
from pathlib import Path

import h5py
from pydantic import BaseModel, validator

# initialize logger
logger = logging.getLogger(__name__)


class FilePath(BaseModel):
    """Pydantic class for validating file paths.

    It ensures that the file path:
    - is, or can be converted to, a pathlib Path object
    - indeed points to a file
    """

    path: Path

    @validator("path")
    def file_must_exist(cls, value):
        if not value.is_file():
            error_msg = f"File not found: {value}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        return value


class DeepLabCutPosesFile(FilePath):
    """Pydantic class for validating file paths containing
    pose estimation results from DeepLabCut (DLC).

    In addition to the checks performed by the FilePath class,
    this class also checks that the file has one of the two
    expected suffixes - ".h5" or ".csv".
    """

    @validator("path")
    def file_must_have_valid_suffix(cls, value):
        if value.suffix not in (".h5", ".csv"):
            error_msg = (
                "Expected a file with pose estimation results from "
                "DeepLabCut, in one of '.h5' or '.csv' formats. "
                f"Received a file with suffix '{value.suffix}' instead."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        return value

    @validator("path")
    def h5_file_must_contain_expected_df(cls, value):
        if value.suffix == ".h5":
            with h5py.File(value, "r") as f:
                dataset = "df_with_missing"
                if dataset not in list(f.keys()):
                    error_msg = (
                        f"Expected dataset '{dataset}' not found in file: "
                        f"{value}. Ensure this is a valid DLC output file."
                    )
                    logger.error(error_msg)
                    raise ValueError(error_msg)
        return value


class SleapPosesFile(FilePath):
    """Pydantic class for validating file paths containing
    analysis results from SLEAP.

    In addition to the checks performed by the FilePath class,
    this class also ensures that:
    - the file has the expected suffix ".h5",
    - the file contains some expected datasets
    - the dataset array shapes are consistent with each other
    """

    @validator("path")
    def file_must_have_valid_suffix(cls, value):
        if value.suffix != ".h5":
            error_msg = (
                "Expected a SLEAP analysis file in '.h5' format. "
                f"Received a file with suffix '{value.suffix}' instead."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        return value

    @validator("path")
    def file_must_contain_expected_datasets(cls, value):
        expected_datasets = ["tracks", "track_occupancy", "node_names"]
        with h5py.File(value, "r") as f:
            for dataset in expected_datasets:
                if dataset not in list(f.keys()):
                    error_msg = (
                        f"Expected dataset '{dataset}' not found in file: "
                        f"{value}. Ensure this is a valid SLEAP analysis file."
                    )
                    logger.error(error_msg)
                    raise ValueError(error_msg)
        return value

    @validator("path")
    def dataset_shapes_must_be_consistent(cls, value):
        with h5py.File(value, "r") as f:
            n_tracks, n_dims, n_nodes, n_frames = f["tracks"].shape
            node_names = [n.decode() for n in f["node_names"][:]]
            track_names = [n.decode() for n in f["track_names"][:]]

            try:
                assert f["track_occupancy"].shape == (n_frames, n_tracks)
                assert n_nodes == len(node_names)
                assert n_tracks == len(track_names)
            except AssertionError:
                error_msg = (
                    f"Dataset shapes in file {value} are inconsistent. "
                    f"Ensure this is a valid SLEAP analysis file."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

        return value
