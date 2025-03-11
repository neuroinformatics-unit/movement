import pytest
import xarray as xr
from loguru import logger as loguru_logger

from movement.utils.logging import MovementLogger, log_to_attrs, logger

log_methods = ["debug", "info", "warning", "error", "exception"]


@pytest.mark.parametrize("method", log_methods)
def test_log_to_file(method):
    """Ensure the correct logger method is called and
    the expected message is in the logfile.
    """
    log_method = getattr(logger, method)
    log_message = f"{method} message"
    log_method(log_message)
    with open(pytest.LOG_FILE) as f:
        all_lines = f.readlines()
    # For exceptions, the last line is the traceback
    last_line = all_lines[-1] if method != "exception" else all_lines[-2]
    level = method.upper() if method != "exception" else "ERROR"
    assert level in last_line
    assert log_message in last_line


def test_logger_repr():
    """Ensure the custom logger's representation equals the loguru logger."""
    assert print(MovementLogger()) == print(loguru_logger)


@pytest.mark.parametrize(
    "input_data",
    ["valid_poses_dataset", "valid_bboxes_dataset"],
)
@pytest.mark.parametrize(
    "selector_fn, expected_selector_type",
    [
        (lambda ds: ds, xr.Dataset),  # take full dataset
        (lambda ds: ds.position, xr.DataArray),  # take position data array
    ],
)
def test_log_to_attrs(
    input_data, selector_fn, expected_selector_type, request
):
    """Test that the ``log_to_attrs()`` decorator appends
    log entries to the dataset's or the data array's ``log``
    attribute and check that ``attrs`` contains all the expected values.
    """

    # a fake operation on the dataset to log
    @log_to_attrs
    def fake_func(data, arg, kwarg=None):
        return data

    # apply operation to dataset or data array
    dataset = request.getfixturevalue(input_data)
    input_data = selector_fn(dataset)
    output_data = fake_func(input_data, "test1", kwarg="test2")

    # check the log in the dataset is as expected
    assert isinstance(output_data, expected_selector_type)
    assert "log" in output_data.attrs
    assert output_data.attrs["log"][0]["operation"] == "fake_func"
    assert output_data.attrs["log"][0]["arg_1"] == "test1"
    assert output_data.attrs["log"][0]["kwarg"] == "test2"
