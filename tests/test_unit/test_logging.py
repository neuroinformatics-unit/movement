import json
import warnings

import pytest
import xarray as xr
from loguru import logger as loguru_logger

from movement.utils.logging import (
    MovementLogger,
    log_to_attrs,
    logger,
    showwarning,
)

log_methods = ["debug", "info", "warning", "error", "exception"]


def assert_log_entry_in_file(expected_components, log_file):
    """Assert that a log entry with the expected components is
    found in the log file.
    """
    with open(log_file) as f:
        all_lines = f.readlines()
    assert any(
        all(component in line for component in expected_components)
        for line in all_lines
    ), (
        f"Expected log entry with components {expected_components} "
        "not found in log file."
    )


@pytest.mark.parametrize("method", log_methods)
def test_log_to_file(method):
    """Ensure the correct logger method is called and
    the expected message is in the logfile.
    """
    log_method = getattr(logger, method)
    log_message = f"{method} message"
    log_method(log_message)
    level = method.upper() if method != "exception" else "ERROR"
    # Check if a matching log entry is found in the log file
    assert_log_entry_in_file([level, log_message], pytest.LOG_FILE)


def test_showwarning():
    """Ensure the custom ``showwarning`` function is called when a
    warning is issued.
    """
    kwargs = {
        "message": "This is a deprecation warning",
        "category": DeprecationWarning,
        "stacklevel": 2,
    }
    warnings.showwarning = showwarning
    warnings.warn(**kwargs)
    # Check if the warning message is in the log file
    expected_components = [kwargs["category"].__name__, kwargs["message"]]
    assert_log_entry_in_file(expected_components, pytest.LOG_FILE)


def test_logger_repr():
    """Ensure the custom logger's representation equals the loguru logger."""
    assert repr(MovementLogger()) == repr(loguru_logger)


@pytest.mark.parametrize(
    "input_data",
    ["valid_poses_dataset", "valid_bboxes_dataset"],
)
@pytest.mark.parametrize(
    "selector_fn, expected_selector_type",
    [
        (lambda ds: ds, xr.Dataset),  # take full dataset
        (lambda ds: ds.position, xr.DataArray),  # take position DataArray
    ],
)
@pytest.mark.parametrize(
    "extra_kwargs",
    [{}, {"extra1": 42}],
    ids=["no_extra_kwargs", "with_extra_kwargs"],
)
def test_log_to_attrs(
    input_data, selector_fn, expected_selector_type, extra_kwargs, request
):
    """Test that the ``log_to_attrs()`` decorator saves
    log entries to the dataset's or data array's ``log``
    attribute.
    """

    @log_to_attrs
    def fake_func(data, arg, kwarg=None, **kwargs):
        return data

    # Apply operation to dataset or data array
    dataset = request.getfixturevalue(input_data)
    input_data = selector_fn(dataset)
    output_data = fake_func(input_data, "test1", kwarg="test2", **extra_kwargs)

    # Check that output is as expected
    assert isinstance(output_data, expected_selector_type)
    assert "log" in output_data.attrs

    # Deserialize the log from JSON
    log_entries = json.loads(output_data.attrs["log"])
    assert isinstance(log_entries, list)
    assert len(log_entries) == 1

    log_entry = log_entries[0]
    assert log_entry["operation"] == "fake_func"
    assert log_entry["arg"] == "'test1'"  # repr() puts quotes around strings
    if extra_kwargs:
        assert log_entry["kwargs"] == "{'extra1': 42}"
    else:
        assert "kwargs" not in log_entry


def test_log_to_attrs_json_decode_error(valid_poses_dataset):
    """Test that a JSON decode error in the log attribute is handled."""

    @log_to_attrs
    def fake_func(data):
        return data

    # Create a dataset with an invalid log attribute
    invalid_log = '[{"invalid_json": "missing_quote}]'  # Invalid JSON
    valid_poses_dataset.attrs["log"] = invalid_log

    # Call the function to trigger the decorator
    result = fake_func(valid_poses_dataset)

    # Check that a warning is written to the log file
    assert_log_entry_in_file(
        ["WARNING", "Failed to decode existing log in attributes"],
        pytest.LOG_FILE,
    )

    # Check that the log contains only the new entry from the fake_func call
    log_entries = json.loads(result.attrs["log"])
    assert len(log_entries) == 1
    assert log_entries[0]["operation"] == "fake_func"
