import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from rich.logging import RichHandler

FORMAT = (
    "%(asctime)s - %(levelname)s - "
    "%(processName)s %(filename)s:%(lineno)s - %(message)s"
)


def configure_logging(
    log_level: int = logging.INFO,
    logger_name: Optional[str] = None,
    log_to_console: bool = False,
):
    """Configure the logging module.
    This function sets up a circular log file with a rotating file handler.

    Parameters
    ----------
    log_level : int, optional
        The logging level to use. Defaults to logging.INFO.
    logger_name : str, optional
        The name of the logger to configure. If None (default),
        the name of the calling package is used.
    log_to_console : bool
        Whether to log to the console in addition to the log file.
        Defaults to False.
    """

    if logger_name is None:
        # Get the name of the calling package
        logger_name = __name__.split(".")[0]

    # If a logger with the given name is already configured
    if logger_name in logging.root.manager.loggerDict:
        # skip configuration
        return

    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)

    # Set the log directory and file path
    log_directory = Path.home() / ".movement"
    log_directory.mkdir(parents=True, exist_ok=True)
    log_file = log_directory / f"{logger_name}.log"

    # Create a rotating file handler
    max_log_size = 1 * 1024 * 1024  # 1 MB
    backup_count = 5  # Number of archived log files to keep
    handler = RotatingFileHandler(
        log_file, maxBytes=max_log_size, backupCount=backup_count
    )

    # Create a formatter and set it to the handler
    formatter = logging.Formatter(FORMAT)
    handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(handler)

    if log_to_console:
        # Also log to the console using RichHandler
        logger.addHandler(RichHandler())
