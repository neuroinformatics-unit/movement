import logging

import pytest

from movement.logging import log_error, log_warning

log_messages = {
    "DEBUG": "This is a debug message",
    "INFO": "This is an info message",
    "WARNING": "This is a warning message",
    "ERROR": "This is an error message",
}


@pytest.mark.parametrize("level, message", log_messages.items())
def test_logfile_contains_message(level, message):
    """Check if the last line of the logfile contains
    the expected message."""
    logger = logging.getLogger("movement")
    eval(f"logger.{level.lower()}('{message}')")
    log_file = logger.handlers[0].baseFilename
    with open(log_file, "r") as f:
        last_line = f.readlines()[-1]
    assert level in last_line
    assert message in last_line


def test_log_error(caplog):
    """Check if the log_error function
    logs the error message and returns an Exception."""
    with pytest.raises(ValueError):
        raise log_error(ValueError, "This is a test error")
    assert caplog.records[0].message == "This is a test error"
    assert caplog.records[0].levelname == "ERROR"


def test_log_warning(caplog):
    """Check if the log_warning function
    logs the warning message."""
    log_warning("This is a test warning")
    assert caplog.records[0].message == "This is a test warning"
    assert caplog.records[0].levelname == "WARNING"
