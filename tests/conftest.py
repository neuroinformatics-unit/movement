"""Fixtures and configurations shared by the entire test suite."""

from glob import glob

import numpy as np
import pytest
from _pytest.logging import LogCaptureFixture

from movement.sample_data import fetch_dataset_paths, list_datasets
from movement.utils.logging import logger


def _to_module_string(path: str) -> str:
    """Convert a file path to a module string."""
    return path.replace("/", ".").replace("\\", ".").replace(".py", "")


pytest_plugins = [
    _to_module_string(fixture)
    for fixture in glob("tests/fixtures/*.py")
    if "__" not in fixture
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
