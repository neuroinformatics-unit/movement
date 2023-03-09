from pathlib import Path
from typing import Union, Optional
import pandas as pd


def validate_file_path(
    filepath: Union[Path, str],
    suffix: Optional[str] = None
) -> Path:
    """Validate a file path.

    This function ensures that the `filepath` is a Path object,
    that it leads to an existing file, and that it has the correct
    extension if `suffix` is provided.

    Parameters
    ----------
    filepath : pathlib Path or str
        Path to the file.
    suffix : str or None, optional
        Expected file extension including the dot (e.g. ".h5").
        The default is None.

    Returns
    -------
    pathlib Path
        Path to the file.

    Raises
    ------
    TypeError
        If `filepath` is not a Path object or a string.
    FileNotFoundError
        If `filepath` does not lead to an existing file.
    ValueError
        If `filepath` does not have the expected extension.
    """

    # Ensure filepath is a Path object or a string
    if not isinstance(filepath, (Path, str)):
        raise TypeError(f"Filepath must be a Path object or a str. "
                        f"Received {type(filepath)} instead.")
    # Ensure a Path object
    filepath = Path(filepath)

    # Ensure filepath leads to an existing file
    if not filepath.is_file():
        raise FileNotFoundError(f"File not found: {filepath}")

    if suffix is not None:
        if filepath.suffix != suffix:
            raise ValueError(
                f"File must have the extension {suffix}. "
                f"Received {filepath} instead."
            )

    return filepath


def validate_dataframe(df: object) -> pd.DataFrame:
    """ This function ensures that `df` is
    a non-empty pandas DataFrame.

    Parameters
    ----------
    df : object
        Object to validate.

    Raises
    ------
    TypeError
        If `df` is not a pandas DataFrame.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame. "
                        f"Received {type(df)} instead.")

    return df
