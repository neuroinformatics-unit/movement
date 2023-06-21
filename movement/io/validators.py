import logging
from pathlib import Path

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

    @validator("path", pre=True)  # runs before other validators
    def convert_to_path(cls, value):
        return Path(value)

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


class SleapPredictionsFile(FilePath):
    """Pydantic class for validating file paths containing
    pose estimation results (predictions) from SLEAP.

    In addition to the checks performed by the FilePath class,
    this class also checks that the file has one of the two
    expected suffixes - ".slp" or ".h5".
    """

    @validator("path")
    def file_must_have_valid_suffix(cls, value):
        if value.suffix not in (".slp", ".h5"):
            error_msg = (
                "Expected a file with pose estimation results (predictions)"
                "from SLEAP, in one of '.slp' or '.h5' formats. "
                f"Received a file with suffix '{value.suffix}' instead."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        return value
