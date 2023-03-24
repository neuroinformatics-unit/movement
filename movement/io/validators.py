from pathlib import Path

from pydantic import BaseModel, validator


class DeeplabcutPosesFile(BaseModel):
    """Validates files containing pose estimation results from
    Deeplabcut (DLC).

    Pydantic will automatically enforce input data types and
    raise errors if the data is invalid.
    We also run some additional checks on top of that:
     - the file must exist
     - the file must have the '.h5' suffix
    """

    file_path: Path

    @validator("file_path")
    def file_must_exist(cls, file_path):
        file_path = Path(file_path)
        if not file_path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")
        return file_path

    @validator("file_path")
    def file_path_must_have_h5_suffix(cls, file_path):
        if file_path.suffix != ".h5":
            raise ValueError(
                f"File must have the '.h5' suffix. "
                f"Received {file_path} instead."
            )
        return file_path
