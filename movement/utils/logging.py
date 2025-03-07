"""Logging utilities for the movement package."""

import sys
from collections.abc import Callable
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
    """Configure a rotating log file.

    This function sets up a rotating log file that logs at the DEBUG level
    with a maximum size of 5 MB and retains the last 5 log files.
    This is in addition to the default loguru logging to ``sys.stderr``.

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
    # Add a rotating file handler
    logger.add(
        log_file, level="DEBUG", format=FORMAT, rotation="5 MB", retention=5
    )
    return log_file


def log_redirect(func: Callable):
    """Redirect logging calls to the appropriate logger method.

    This decorator is used to redirect a function in the format
    `log_<method>` to the appropriate ``loguru logger`` method
    (e.g. `log_info` to `logger.info`) and log the ``message``.
    The function should accept either one or two arguments.
    If one argument is passed, it is assumed to be the ``message`` to log.
    If two arguments are passed, the first is assumed to be an
    Exception and the second is the ``message`` to log. This
    Exception will be returned if the method is ``"error"`` or
    ``"exception"``.

    Parameters
    ----------
    func : Callable
        The function to be decorated.
        The function name must be in the format `log_<method>`.

    Returns
    -------
    Callable
        The decorated function.

    Raises
    ------
    ValueError
        If the number of arguments passed to
        the decorated function is not 1 or 2.

    Example
    -------
    >>> @log_redirect
    ... def log_info(message: str):
    ...     pass
    >>> log_info("This is an info message.")
    >>> @log_redirect
    ... def log_error(exception: Exception, message: str):
    ...     return exception(message)
    >>> log_error(ValueError, "This is an error message.")

    """

    def wrapper(*args, **kwargs):
        method = func.__name__.split("_")[1]
        if len(args) == 1:
            message = args[0]
            exception = None
        elif len(args) == 2:
            exception = args[0]
            message = args[1]
        else:
            raise ValueError("Invalid number of arguments. Must be 1 or 2.")
        getattr(logger, method)(message)
        if exception and method in ["error", "exception"]:
            return exception(message)

    return wrapper


# Dynamically create log functions
log_methods = ["debug", "info", "warning", "error", "exception"]
for method in log_methods:
    func_name = f"log_{method}"

    # Create a new function
    def log_function(*args, **kwargs):  # noqa: D103 # pragma: no cover
        pass

    # Change the name of the function
    log_function.__name__ = func_name
    setattr(sys.modules[__name__], func_name, log_redirect(log_function))


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
