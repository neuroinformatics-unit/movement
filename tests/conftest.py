import logging

import pytest

from movement.log_config import configure_logging


@pytest.fixture(autouse=True)
def setup_logging(tmp_path):
    """Set up logging for the test module.
    Redirects all logging to a temporary directory."""
    configure_logging(
        log_level=logging.DEBUG,
        logger_name="movement",
        log_directory=(tmp_path / ".movement"),
    )
