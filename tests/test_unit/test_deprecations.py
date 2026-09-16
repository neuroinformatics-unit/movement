"""Tests for deprecated API elements (functions, methods, classes)."""

from contextlib import nullcontext

import pytest
import xarray as xr

from movement.io import load, save_poses


@pytest.mark.parametrize(
    "deprecated_callable, mocked_inputs, patch_context, check_in_message",
    [
        (
            load.rename_legacy_dimensions,
            lambda request: {"ds": xr.Dataset()},
            nullcontext(),
            r"`rename_legacy_dimensions` is deprecated",
        ),
        (
            save_poses.to_nwb_file,
            lambda request: {
                "ds": request.getfixturevalue("valid_poses_dataset")
            },
            nullcontext(),
            r"`to_nwb_file` has been renamed to `to_nwb_file_object`",
        ),
    ],
)
def test_deprecated_callable(
    deprecated_callable,
    mocked_inputs,
    patch_context,
    check_in_message,
    request,
):
    """Test that a deprecated callable emits a DeprecationWarning.

    When deprecating a callable API element, add a parametrised case to
    this test. See an older version of this file for concrete examples:
    https://github.com/neuroinformatics-unit/movement/blob/v0.16.0/tests/test_unit/test_deprecations.py

    Parameters
    ----------
    deprecated_callable : callable
        The deprecated callable (function, bound method, or class).
    mocked_inputs : callable
        A function taking the ``request`` fixture and returning a dict of
        keyword arguments to pass to the callable. Wrapped in a lambda so
        that fixtures can be resolved lazily via ``request.getfixturevalue``.
    patch_context : contextlib.AbstractContextManager
        A ``unittest.mock.patch`` context manager that prevents the
        callable from doing real work, or
        ``contextlib.nullcontext()`` if patching is not needed.
    check_in_message : str
        A string or regex that must appear in the warning message
        (typically the name of the recommended replacement).
    request : pytest.FixtureRequest
        Used to resolve fixtures referenced inside ``mocked_inputs``.

    """
    with patch_context, pytest.deprecated_call(match=check_in_message):
        deprecated_callable(**mocked_inputs(request))
