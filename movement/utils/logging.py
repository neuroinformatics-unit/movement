"""Logging utilities for the movement package."""

import sys
from datetime import datetime
from functools import wraps
from pathlib import Path

from loguru import logger

FORMAT = "{time} - {level} - {process.name} {file}:{line} - {message}"
DEFAULT_LOG_DIRECTORY = Path.home() / ".movement"


def configure_logging(
    log_file_name: str = "movement",
    log_directory: Path = DEFAULT_LOG_DIRECTORY,
) -> str:
    """Configure a rotating log file and console (stdout) logger.

    This function sets up a rotating log file that logs at the DEBUG level
    with a maximum size of 5 MB and retains the last 5 log files.
    It also configures a console logger that logs at the INFO level.

    Parameters
    ----------
    log_file_name : str, optional
        The name of the log file. Defaults to "movement".
    log_directory : pathlib.Path, optional
        The directory to store the log file in. Defaults to
        ~/.movement. A different directory can be specified,
        for example for testing purposes.

    Returns
    -------
    str
        The path to the log file.

    """
    # Set the log directory and file path
    log_directory.mkdir(parents=True, exist_ok=True)
    log_file = (log_directory / f"{log_file_name}.log").as_posix()
    # Remove any existing handlers
    logger.remove()
    # Add a console handler and a rotating file handler
    logger.add(sys.stdout, level="INFO", format=FORMAT)
    logger.add(
        log_file, level="DEBUG", format=FORMAT, rotation="5 MB", retention=5
    )
    return log_file


def _log_and_return_exception(log_func, exception, message: str):
    """Log a message and return an exception.

    Parameters
    ----------
    log_func : callable
        The logging function to use
        (e.g., ``logger.error``, ``logger.exception``).
    exception : Exception
        The exception to log and return.
    message : str
        The log message.

    Returns
    -------
    Exception
        The exception that was passed in.

    """
    log_func(message)
    return exception(message)


def log_error(error, message: str):
    """Log an error message and return the Error.

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
    return _log_and_return_exception(logger.error, error, message)


def log_exception(exception, message: str):
    """Log an exception message and return the Exception.

    Parameters
    ----------
    exception : Exception
        The exception to log and return.
    message : str
        The exception message.

    Returns
    -------
    Exception
        The exception that was passed in.

    """
    return _log_and_return_exception(logger.exception, exception, message)


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
