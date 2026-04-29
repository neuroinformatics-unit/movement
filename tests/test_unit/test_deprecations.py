"""Tests for deprecated functions.

When deprecating a function, add a parametrised case to the
``test_deprecation`` test below. Each case should specify:

- ``deprecated_function``: the callable being deprecated.
- ``mocked_inputs``: keyword arguments to pass to the function.
- ``patch_context``: a ``unittest.mock.patch`` context (or
  ``contextlib.nullcontext()`` if patching is not needed) that
  prevents the function from doing real work.
- ``check_in_message``: strings that must appear in the warning
  message (typically the names of the recommended replacements).

See the git history of this file for concrete examples.
"""

import pytest


@pytest.mark.parametrize(
    "deprecated_function, mocked_inputs, patch_context, check_in_message",
    [],
)
def test_deprecation(
    deprecated_function, mocked_inputs, patch_context, check_in_message
):
    """Test that calling a deprecated function raises a DeprecationWarning
    with the expected message.
    """
    with patch_context, pytest.warns(DeprecationWarning) as record:
        _ = deprecated_function(**mocked_inputs)
    assert f"{deprecated_function.__name__}` is deprecated" in str(
        record[0].message
    )
    assert all(
        message in str(record[0].message) for message in check_in_message
    )
