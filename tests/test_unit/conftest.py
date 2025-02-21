from collections.abc import Callable

import numpy as np
import pytest
import xarray as xr


@pytest.fixture(scope="session")
def push_into_range() -> Callable[
    [xr.DataArray | np.ndarray, float, float], xr.DataArray | np.ndarray
]:
    """Return a function for wrapping angles.

    This is a factory fixture that returns a method for wrapping angles
    into a user-specified range.
    """

    def _push_into_range(
        numeric_values: xr.DataArray | np.ndarray,
        lower: float = -180.0,
        upper: float = 180.0,
    ) -> xr.DataArray | np.ndarray:
        """Coerce values into the range (lower, upper].

        Primarily used to wrap returned angles into a particular range,
        such as (-pi, pi].

        The interval width is the value ``upper - lower``.
        Each element in ``values`` that starts less than or equal to the
        ``lower`` bound has multiples of the interval width added to it,
        until the result lies in the desirable interval.

        Each element in ``values`` that starts greater than the ``upper``
        bound has multiples of the interval width subtracted from it,
        until the result lies in the desired interval.
        """
        translated_values = (
            numeric_values.values.copy()
            if isinstance(numeric_values, xr.DataArray)
            else numeric_values.copy()
        )

        interval_width = upper - lower
        if interval_width <= 0:
            raise ValueError(
                f"Upper bound ({upper}) must be strictly "
                f"greater than lower bound ({lower})"
            )

        while np.any(
            (translated_values <= lower) | (translated_values > upper)
        ):
            translated_values[translated_values <= lower] += interval_width
            translated_values[translated_values > upper] -= interval_width

        if isinstance(numeric_values, xr.DataArray):
            translated_values = numeric_values.copy(
                deep=True, data=translated_values
            )
        return translated_values

    return _push_into_range
