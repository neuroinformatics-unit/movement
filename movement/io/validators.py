import logging
from pathlib import Path

from pydantic import BaseModel, validator

# initialize logger
logger = logging.getLogger(__name__)


class DeeplabcutPosesFile(BaseModel):
    """Pydantic class for validating files containing
    pose estimation results from Deeplabcut (DLC).

    Pydantic will enforce the input data type.
    This class additionally checks that the file exists
    and has a valid suffix.
    """

    file_path: Path

    @validator("file_path", pre=True)  # runs before other validators
    def convert_to_path(cls, value):
        return Path(value)

    @validator("file_path")
    def file_must_exist(cls, value):
        if not value.is_file():
            error_msg = f"File not found: {value}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        return value

    @validator("file_path")
    def file_must_have_valid_suffix(cls, value):
        if value.suffix not in (".h5", ".csv"):
            error_msg = (
                f"File suffix must be '.h5' or '.csv'. "
                f"Received `{value.suffix}` instead."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        return value