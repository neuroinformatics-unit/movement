"""Wrapper for broadcasting operations across xarray dimensions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Concatenate, ParamSpec, TypeVar

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import ArrayLike
    from xarray import DataArray

Scalar = TypeVar("Scalar")
KeywordArgs = ParamSpec("KeywordArgs")


def make_broadcastable(
    f: Callable[
        Concatenate[ArrayLike, KeywordArgs],
        Scalar,
    ],
) -> Callable[Concatenate[DataArray, KeywordArgs], DataArray]:
    """Broadcast a 1D function along a ``xr.DataArray`` dimension.

    Parameters
    ----------
    f : Callable
        1D function to be converted into a broadcast-able function.
        ``f`` should be callable as ``f([x, y, ...], *args, **kwargs), and
        return a scalar variable.

    Returns
    -------
    Callable
        Callable with ``data, *args, broadcast_dimension = str, **kwargs)``,
        that applies ``f`` along the ``broadcast_dimension`` of ``data``.
        ``args`` and ``kwargs`` match those passed to ``f``, and retain the
        same interpretations.

    """

    def f_along_axis(
        data: ArrayLike, axis: int, *f_args, **f_kwargs
    ) -> np.ndarray:
        return np.apply_along_axis(f, axis, data, *f_args, **f_kwargs)

    def inner(
        data: DataArray,
        *args: KeywordArgs.args,
        broadcast_dimension: str = "space",
        **kwargs: KeywordArgs.kwargs,
    ) -> DataArray:
        return data.reduce(f_along_axis, broadcast_dimension, *args, **kwargs)

    return inner


def make_broadcastable_over_space(
    f: Callable[
        Concatenate[ArrayLike, KeywordArgs],
        Scalar,
    ],
) -> Callable[Concatenate[DataArray, KeywordArgs], DataArray]:
    """Broadcast a 1D function along a ``xr.DataArray`` ``"space"`` dimension.

    Parameters
    ----------
    f : Callable
        1D function to be converted into a broadcast-able function.
        ``f`` should be callable as ``f([x, y, ...], *args, **kwargs), and
        return a scalar variable.

    Returns
    -------
    Callable
        Callable with ``data, *args, **kwargs)``, that applies ``f`` along the
        ``broadcast_dimension`` of ``data``. ``args`` and ``kwargs`` match
        those passed to ``f``, and retain the same interpretations.

    Notes
    -----
    This is a convenience alias for

    ```python
    def(data, *args, **kwargs):
        return make_broadcastable(f)(
            data,
            *args,
            broadcast_dimension = "space",
            **kwargs,
        )
    ```

    and is primarily useful when we want to write a function that acts on
    coordinates, that needs to be cast across one dimension of an
    ``xarray.DataArray``.

    """
    decorate = make_broadcastable(f)

    def inner(
        data: DataArray, *args: KeywordArgs.args, **kwargs: KeywordArgs.kwargs
    ) -> DataArray:
        return decorate(
            data,
            *args,
            **kwargs,
            broadcast_dimension="space",  # type: ignore
        )

    return inner
