import logging
from logging.handlers import RotatingFileHandler

FORMAT = "%(asctime)s - %(levelname)s - "
FORMAT += "%(processName)s %(filename)s:%(lineno)s - %(message)s"


def configure_logging():
    """Configure the logging module."""

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    log_file = "movement.log"

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
