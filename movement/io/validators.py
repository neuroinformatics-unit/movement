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

    filepath: Path

    @validator("filepath")
    def file_must_exist(cls, filepath):
        filepath = Path(filepath)
        if not filepath.is_file():
            raise FileNotFoundError(f"File not found: {filepath}")
        return filepath

    @validator("filepath")
    def filepath_must_have_h5_suffix(cls, filepath):
        if filepath.suffix != ".h5":
            raise ValueError(
                f"File must have the '.h5' suffix. "
                f"Received {filepath} instead."
            )
        return filepath
