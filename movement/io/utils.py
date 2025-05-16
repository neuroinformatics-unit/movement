"""Functions shared across the ``movement`` IO module."""

from pathlib import Path

from movement.utils.logging import logger
from movement.validators.files import ValidFile


def _validate_file_path(
    file_path: str | Path, expected_suffix: list[str]
) -> ValidFile:
    """Validate the input file path.

    We check that the file has write permission and the expected suffix(es).

    Parameters
    ----------
    file_path : pathlib.Path or str
        Path to the file to validate.
    expected_suffix : list of str
        Expected suffix(es) for the file.

    Returns
    -------
    ValidFile
        The validated file.

    Raises
    ------
    OSError
        If the file cannot be written.
    ValueError
        If the file does not have the expected suffix.

    """
    try:
        file = ValidFile(
            file_path,
            expected_permission="w",
            expected_suffix=expected_suffix,
        )
    except (OSError, ValueError) as error:
        logger.error(error)
        raise
    return file
