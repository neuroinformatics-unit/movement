"""Tests for deprecated API elements (functions, methods, classes)."""

import pytest


@pytest.mark.parametrize(
    "deprecated_callable, mocked_inputs, patch_context, check_in_message",
    [],
)
def test_deprecated_callable(
    deprecated_callable, mocked_inputs, patch_context, check_in_message
):
    """Test that a deprecated callable emits a DeprecationWarning.

    When deprecating a callable API element, add a parametrised case to
    this test. See an older version of this file for concrete examples:
    https://github.com/neuroinformatics-unit/movement/blob/v0.16.0/tests/test_unit/test_deprecations.py

    Parameters
    ----------
    deprecated_callable : callable
        The deprecated callable (function, bound method, or class).
    mocked_inputs : dict
        Keyword arguments to pass to the callable.
    patch_context : contextlib.AbstractContextManager
        A ``unittest.mock.patch`` context manager that prevents the
        callable from doing real work, or
        ``contextlib.nullcontext()`` if patching is not needed.
    check_in_message : list of str
        Strings that must appear in the warning message (typically the
        names of the recommended replacements).

    """
    with patch_context, pytest.warns(DeprecationWarning) as record:
        _ = deprecated_callable(**mocked_inputs)
    assert f"{deprecated_callable.__name__}` is deprecated" in str(
        record[0].message
    )
    assert all(
        message in str(record[0].message) for message in check_in_message
    )
