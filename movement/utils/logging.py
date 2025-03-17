"""Logging utilities for the movement package."""

import sys
from datetime import datetime
from functools import wraps
from pathlib import Path

from loguru import logger as loguru_logger

DEFAULT_LOG_DIRECTORY = Path.home() / ".movement"


class MovementLogger:
    """A custom logger extending the loguru logger."""

    def __init__(self):
        """Initialize the logger with the loguru logger."""
        self.logger = loguru_logger

    def configure(
        self,
        log_file_name: str = "movement",
        log_directory: Path = DEFAULT_LOG_DIRECTORY,
        console: bool = True,
    ):
        """Configure a rotating file logger and optionally a console logger.

        This method configures a rotating log file that
        logs at the DEBUG level with a maximum size of 5 MB
        and retains the last 5 log files.
        It also optionally adds a console (``sys.stderr``) handler
        that logs at the WARNING level.

        Parameters
        ----------
        log_file_name : str, optional
            The name of the log file. Defaults to ``"movement"``.
        log_directory : pathlib.Path, optional
            The directory to store the log file in. Defaults to
            ``"~/.movement"``. A different directory can be specified,
            for example, for testing purposes.
        console : bool, optional
            Whether to add a console logger. Defaults to ``True``.

        """
        log_directory.mkdir(parents=True, exist_ok=True)
        log_file = (log_directory / f"{log_file_name}.log").as_posix()
        self.remove()
        if console:
            self.add(sys.stderr, level="WARNING")
        self.add(log_file, level="DEBUG", rotation="5 MB", retention=5)
        return log_file

    def _log_and_return_exception(self, log_method, message, *args, **kwargs):
        """Log the message and return an Exception if specified."""
        log_method(message, *args, **kwargs)
        if isinstance(message, Exception):
            return message

    def error(self, message, *args, **kwargs):
        """Override the error method to optionally return an Exception."""
        return self._log_and_return_exception(
            self.logger.error, message, *args, **kwargs
        )

    def exception(self, message, *args, **kwargs):
        """Override the exception method to optionally return an exception."""
        return self._log_and_return_exception(
            self.logger.exception, message, *args, **kwargs
        )

    def __getattr__(self, name):
        """Redirect attribute access to the loguru logger."""
        return getattr(self.logger, name)

    def __repr__(self):
        """Return the loguru logger's representation."""
        return repr(self.logger)


logger = MovementLogger()


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
