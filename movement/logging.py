import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

FORMAT = (
    "%(asctime)s - %(levelname)s - "
    "%(processName)s %(filename)s:%(lineno)s - %(message)s"
)


def configure_logging(
    log_level: int = logging.DEBUG,
    logger_name: str = "movement",
    log_directory: Path = Path.home() / ".movement",
):
    """Configure the logging module.
    This function sets up a circular log file with a rotating file handler.

    Parameters
    ----------
    log_level : int, optional
        The logging level to use. Defaults to logging.INFO.
    logger_name : str, optional
        The name of the logger to configure.
        Defaults to "movement".
    log_directory : pathlib.Path, optional
        The directory to store the log file in. Defaults to
        ~/.movement. A different directory can be specified,
        for example for testing purposes.
    """

    # Set the log directory and file path
    log_directory.mkdir(parents=True, exist_ok=True)
    log_file = (log_directory / f"{logger_name}.log").as_posix()

    # Check if logger with the given name is already configured
    logger_configured = logger_name in logging.root.manager.loggerDict

    logger = logging.getLogger(logger_name)

    # Logger needs to be (re)configured if unconfigured or
    # if configured but the log file path has changed
    configure_logger = (
        not logger_configured
        or log_file != logger.handlers[0].baseFilename  # type: ignore
    )

    if configure_logger:
        if logger_configured:
            # remove the handlers to allow for reconfiguration
            logger.handlers.clear()

        logger.setLevel(log_level)

        # Create a rotating file handler
        max_log_size = 5 * 1024 * 1024  # 5 MB
        handler = RotatingFileHandler(log_file, maxBytes=max_log_size)

        # Create a formatter and set it to the handler
        formatter = logging.Formatter(FORMAT)
        handler.setFormatter(formatter)

        # Add the handler to the logger
        logger.addHandler(handler)


def log_error(error, message: str, logger_name: str = "movement"):
    """Log an error message and return the Exception.

    Parameters
    ----------
    error : Exception
        The error to log and return.
    message : str
        The error message.
    logger_name : str, optional
        The name of the logger to use. Defaults to "movement".

    Returns
    -------
    Exception
        The error that was passed in.
    """
    logger = logging.getLogger(logger_name)
    logger.error(message)
    return error(message)


def log_warning(message: str, logger_name: str = "movement"):
    """Log a warning message.

    Parameters
    ----------
    message : str
        The warning message.
    logger_name : str, optional
        The name of the logger to use. Defaults to "movement".
    """
    logger = logging.getLogger(logger_name)
    logger.warning(message)
