import pytest
import xarray as xr
from loguru import logger

from movement.utils.logging import log_error, log_to_attrs, log_warning

log_messages = {
    "DEBUG": "This is a debug message",
    "INFO": "This is an info message",
    "WARNING": "This is a warning message",
    "ERROR": "This is an error message",
}


@pytest.mark.parametrize("level, message", log_messages.items())
def test_logfile_contains_message(level, message, setup_logging):
    """Check if the last line of the logfile contains
    the expected message.
    """
    log_method = getattr(logger, level.lower())
    log_method(message)
    with open(setup_logging) as f:
        last_line = f.readlines()[-1]
    assert level in last_line
    assert message in last_line


def test_log_error(caplog):
    """Check if the log_error function
    logs the error message and returns an Exception.
    """
    with pytest.raises(ValueError):
        raise log_error(ValueError, "This is a test error")
    assert caplog.records[0].message == "This is a test error"
    assert caplog.records[0].levelname == "ERROR"


def test_log_warning(caplog):
    """Check if the log_warning function
    logs the warning message.
    """
    log_warning("This is a test warning")
    assert caplog.records[0].message == "This is a test warning"
    assert caplog.records[0].levelname == "WARNING"


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
