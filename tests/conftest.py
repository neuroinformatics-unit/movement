"""Fixtures and configurations shared by the entire test suite."""

from glob import glob

import pytest
from _pytest.logging import LogCaptureFixture
from loguru import logger

from movement.sample_data import fetch_dataset_paths, list_datasets
from movement.utils.logging import configure_logging


def _to_module_string(path: str) -> str:
    """Convert a file path to a module string."""
    return path.replace("/", ".").replace("\\", ".").replace(".py", "")


pytest_plugins = [
    _to_module_string(fixture)
    for fixture in glob("tests/fixtures/*.py")
    if "__" not in fixture
]


def pytest_configure():
    """Perform initial configuration for pytest.
    Fetches pose data file paths as a dictionary for tests.
    """
    pytest.DATA_PATHS = {}
    for file_name in list_datasets():
        paths_dict = fetch_dataset_paths(file_name)
        data_path = paths_dict.get("poses") or paths_dict.get("bboxes")
        pytest.DATA_PATHS[file_name] = data_path


@pytest.fixture(autouse=True)
def setup_logging(tmp_path):
    """Set up logging for the test module.
    Redirects all logging to a temporary directory.
    """
    configure_logging(
        log_level="DEBUG",
        log_file_name="movement-test",
        log_directory=(tmp_path / ".movement"),
    )


@pytest.fixture
def caplog(caplog: LogCaptureFixture):
    handler_id = logger.add(
        caplog.handler,
        format="{message}",
        level="DEBUG",
        filter=lambda record: record["level"].no >= caplog.handler.level,
        enqueue=False,
    )
    yield caplog
    logger.remove(handler_id)
