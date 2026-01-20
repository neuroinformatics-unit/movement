"""Fixtures and configurations shared by the entire test suite."""

from pathlib import Path

import numpy as np
import pytest
from _pytest.logging import LogCaptureFixture

from movement.sample_data import fetch_dataset_paths, list_datasets
from movement.utils.logging import logger

# define fixtures module path
# by searching relative to this file
# (we assume the structure is always
#  tests.fixtures.<module_name>)
fixtures_dir = Path(__file__).parent / "fixtures"
pytest_plugins = [
    f"tests.fixtures.{fixture.stem}"
    for fixture in fixtures_dir.glob("*.py")
    if "__" not in fixture.name
]


def pytest_sessionstart(session):
    """Set up logging to file and fetch test dataset file paths."""
    # Set up log file in a temporary directory
    tmp_path_factory = session.config._tmp_path_factory
    pytest.LOG_FILE = logger.configure(
        log_file_name=".movement-test",
        log_directory=tmp_path_factory.mktemp(".movement"),
        console=False,
    )
    # Fetch test dataset file paths as a dictionary
    pytest.DATA_PATHS = {}
    for file_name in list_datasets():
        paths_dict = fetch_dataset_paths(file_name)
        data_path = paths_dict.get("poses") or paths_dict.get("bboxes")
        pytest.DATA_PATHS[file_name] = data_path


@pytest.fixture
def caplog(caplog: LogCaptureFixture):
    """Override the caplog fixture by adding a sink
    that propagates loguru to the caplog handler.
    """
    handler_id = logger.add(
        caplog.handler,
        format="{message}",
        level="DEBUG",
        filter=lambda record: record["level"].no >= caplog.handler.level,
        enqueue=False,
    )
    yield caplog
    logger.remove(handler_id)


@pytest.fixture(scope="session")
def rng():
    """Return a random number generator with a fixed seed."""
    return np.random.default_rng(seed=42)
