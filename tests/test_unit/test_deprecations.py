"""Tests for deprecated API elements (functions, methods, classes)."""

from contextlib import nullcontext

import pytest
import xarray as xr

from movement.io import load


@pytest.mark.parametrize(
    "deprecated_callable, mocked_inputs, patch_context, check_in_message",
    [
        (
            load.rename_legacy_dimensions,
            {"ds": xr.Dataset()},
            nullcontext(),
            r"`rename_legacy_dimensions` is deprecated",
        ),
    ],
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
    check_in_message : str
        A string or regex that must appear in the warning message
        (typically the name of the recommended replacement).

    """
    with patch_context, pytest.deprecated_call(match=check_in_message):
        deprecated_callable(**mocked_inputs)
