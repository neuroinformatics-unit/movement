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

    @validator("file_path")
    def file_must_exist(cls, file_path):
        file_path = Path(file_path)
        if not file_path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")
        return file_path

    @validator("file_path")
    def file_must_have_valid_suffix(cls, file_path):
        file_path = Path(file_path)
        if file_path.suffix not in (".h5", ".csv"):
            raise ValueError(
                "File suffix must be '.h5' or '.csv'. "
                f"Received {file_path.suffix} instead."
            )
        return file_path
