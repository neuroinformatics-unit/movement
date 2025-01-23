"""Fixtures and configurations applied to the entire test suite."""

import logging
from glob import glob

import pytest

from movement.sample_data import fetch_dataset_paths, list_datasets
from movement.utils.logging import configure_logging


def to_pytest_plugin_path(string: str) -> str:
    """Convert a file path to a pytest-compatible plugin path."""
    return string.replace("/", ".").replace("\\", ".").replace(".py", "")


pytest_plugins = [
    to_pytest_plugin_path(fixture)
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
        log_level=logging.DEBUG,
        logger_name="movement",
        log_directory=(tmp_path / ".movement"),
    )
