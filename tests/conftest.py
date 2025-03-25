"""Fixtures and configurations shared by the entire test suite."""

import logging
from glob import glob
import re

import pytest

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
        log_level=logging.DEBUG,
        logger_name="movement",
        log_directory=(tmp_path / ".movement"),
    )


def check_error_message(exception_info, expected_pattern):
    """Check that an exception's error message matches the expected pattern.
    
    Parameters
    ----------
    exception_info : ExceptionInfo
        The ExceptionInfo object obtained from pytest.raises context manager.
    expected_pattern : str
        A regex pattern that should match the error message.
        
    Returns
    -------
    bool
        True if the error message matches the pattern, False otherwise.
    """
    return re.search(expected_pattern, str(exception_info.value)) is not None


@pytest.fixture
def check_error():
    """Fixture that provides the check_error_message function."""
    return check_error_message
