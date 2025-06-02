"""Logging utilities for the movement package."""

import inspect
import json
import sys
import warnings
from datetime import datetime
from functools import wraps
from pathlib import Path

from loguru import logger as loguru_logger

DEFAULT_LOG_DIRECTORY = Path.home() / ".movement"


class MovementLogger:
    """A custom logger extending the :mod:`loguru logger <loguru._logger>`."""

    def __init__(self):
        """Initialize the logger with the :mod:`loguru._logger`."""
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
        and retains the last log file.
        It also optionally adds a console (:data:`sys.stderr`) handler
        that logs at the WARNING level.
        Finally, it redirects warnings from the :mod:`warnings` module
        to the logger.

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
        self.add(log_file, level="DEBUG", rotation="5 MB", retention=1)
        # Redirect warnings to the logger
        warnings.showwarning = showwarning
        return log_file

    def _log_and_return_exception(self, log_method, message, *args, **kwargs):
        """Log the message and return an Exception if specified."""
        log_method(message, *args, **kwargs)
        if isinstance(message, Exception):
            return message

    def error(self, message, *args, **kwargs):
        """Log error message and optionally return an Exception.

        This method overrides loguru's
        :meth:`logger.error() <loguru._logger.Logger.error>` to optionally
        return an Exception if the message is an Exception.
        """
        return self._log_and_return_exception(
            self.logger.error, message, *args, **kwargs
        )

    def exception(self, message, *args, **kwargs):
        """Log error message with traceback and optionally return an Exception.

        This method overrides loguru's
        :meth:`logger.exception() <loguru._logger.Logger.exception>` to
        optionally return an Exception if the message is an Exception.
        """
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


def showwarning(message, category, filename, lineno, file=None, line=None):
    """Redirect alerts from the :mod:`warnings` module to the logger.

    This function replaces :func:`logging.captureWarnings` which redirects
    warnings issued by the :mod:`warnings` module to the logging system.
    """
    formatted_message = warnings.formatwarning(
        message, category, filename, lineno, line
    )
    logger.opt(depth=2).warning(formatted_message)


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
        }

        # Extract argument names from the function signature
        signature = inspect.signature(func)
        bound_args = signature.bind(*args, **kwargs)
        bound_args.apply_defaults()

        # Store each argument
        # (excluding the first, which is the Dataset/DataArray itself)
        for param_name, value in list(bound_args.arguments.items())[1:]:
            if param_name == "kwargs" and not value:
                continue  # Skip empty kwargs
            log_entry[param_name] = repr(value)

        if result is not None and hasattr(result, "attrs"):
            log_str = result.attrs.get("log", "[]")
            try:
                log_list = json.loads(log_str)
            except json.JSONDecodeError:
                log_list = []
                logger.warning(
                    f"Failed to decode existing log in attributes: {log_str}. "
                    f"Overwriting with an empty list."
                )

            log_list.append(log_entry)
            result.attrs["log"] = json.dumps(log_list, indent=2)

        return result

    return wrapper
