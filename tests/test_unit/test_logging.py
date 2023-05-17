import logging

import pytest

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
