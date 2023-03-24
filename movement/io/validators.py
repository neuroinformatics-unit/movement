from pathlib import Path

from pydantic import BaseModel, validator


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
            raise FileNotFoundError(f"File not found: {value}")
        return value

    @validator("file_path")
    def file_must_have_valid_suffix(cls, value):
        if value.suffix not in (".h5", ".csv"):
            raise ValueError(
                "File suffix must be '.h5' or '.csv'. "
                f"Received {value.suffix} instead."
            )
        return value
