"""Logging utilities for the movement package."""

from datetime import datetime
from functools import wraps
from pathlib import Path

from loguru import logger

FORMAT = "{time} - {level} - {process.name} {file}:{line} - {message}"
DEFAULT_LOG_DIRECTORY = Path.home() / ".movement"


def configure_logging(
    log_level: str = "DEBUG",
    log_file_name: str = "movement",
    log_directory: Path = DEFAULT_LOG_DIRECTORY,
):
    """Configure a rotating log file for the logger.

    This function sets up a rotating log file for the logger
    with a maximum size of 5 MB and retains the last 5 log files.

    Parameters
    ----------
    log_level : str, optional
        The logging level to use. Defaults to "DEBUG".
    log_file_name : str, optional
        The name of the log file. Defaults to "movement".
    log_directory : pathlib.Path, optional
        The directory to store the log file in. Defaults to
        ~/.movement. A different directory can be specified,
        for example for testing purposes.

    """
    # Set the log directory and file path
    log_directory.mkdir(parents=True, exist_ok=True)
    log_file = (log_directory / f"{log_file_name}.log").as_posix()
    # Add a rotating file handler to the logger
    logger.add(
        log_file, level=log_level, format=FORMAT, rotation="5 MB", retention=5
    )


def log_error(error, message: str):
    """Log an error message and return the Exception.

    Parameters
    ----------
    error : Exception
        The error to log and return.
    message : str
        The error message.

    Returns
    -------
    Exception
        The error that was passed in.

    """
    logger.error(message)
    return error(message)


def log_warning(message: str):
    """Log a warning message.

    Parameters
    ----------
    message : str
        The warning message.

    """
    logger.warning(message)


def log_to_attrs(func):
    """Log the operation performed by the wrapped function.

    This decorator appends log entries to the data's ``log``
    attribute. The wrapped function must accept an :class:`xarray.Dataset`
    or :class:`xarray.DataArray` as its first argument and return an
    object of the same type.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)

        log_entry = {
            "operation": func.__name__,
            "datetime": str(datetime.now()),
            **{f"arg_{i}": arg for i, arg in enumerate(args[1:], start=1)},
            **kwargs,
        }

        # Append the log entry to the result's attributes
        if result is not None and hasattr(result, "attrs"):
            if "log" not in result.attrs:
                result.attrs["log"] = []
            result.attrs["log"].append(log_entry)

        return result

    return wrapper
