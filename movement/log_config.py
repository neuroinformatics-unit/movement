import ctypes
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

FORMAT = "%(asctime)s - %(levelname)s - "
FORMAT += "%(processName)s %(filename)s:%(lineno)s - %(message)s"


def configure_logging(log_level=logging.INFO):
    """Configure the logging module for the current Python script.
    This function sets up a circular log file with a rotating file handler.
    """

    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Set the log directory and file path
    log_directory = Path.home() / ".movement"
    log_directory.mkdir(parents=True, exist_ok=True)
    log_file = log_directory / "movement.log"

    # Make the .movement directory hidden on Windows
    if os.name == "nt":
        FILE_ATTRIBUTE_HIDDEN = 0x02
        try:
            ctypes.windll.kernel32.SetFileAttributesW(
                str(log_directory), FILE_ATTRIBUTE_HIDDEN
            )
        except ctypes.WinError() as e:
            print(f"Unable to hide folder: {e}")

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
