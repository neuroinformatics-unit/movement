import warnings
from pathlib import Path
from typing import Optional

import h5py
import pandas as pd
from pydantic import BaseModel, validator


class DeeplabcutPosesFile(BaseModel):
    """Validates files containing pose estimation results from
    Deeplabcut (DLC).

    Pydantic will automatically enforce input data types and
    raise errors if the data is invalid.
    We also run some additional checks on top of that:
     - the file must exist
     - the file must have the '.h5' suffix
     - if a key is provided, and it is not found in the file,
       a warning is raised and the key is set to None."""

    filepath: Path
    key: Optional[str] = "df_with_missing"

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

    @validator("key")
    def set_key_to_none_if_key_not_in_file(cls, key, values):
        if key is not None:
            with h5py.File(values["filepath"], "r") as h5_file:
                if key not in h5_file:
                    warnings.warn(
                        f"Key '{key}' not found in {values['filepath']}. "
                        "Will try to find a single dataframe in the file."
                    )
                    return None
            return key


def validate_dataframe(df: object) -> pd.DataFrame:
    """Validate that the input object is a non-empty pandas DataFrame.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame to validate.

    Returns
    -------
    pandas DataFrame
        Validated DataFrame.

    Raises
    ------
    TypeError
        If the input is not a pandas DataFrame.
    ValueError
        If the DataFrame is empty.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"Expected a pandas DataFrame. Found {type(df)} instead."
        )
    if df.empty:
        raise ValueError("DataFrame cannot be empty.")
    return df
