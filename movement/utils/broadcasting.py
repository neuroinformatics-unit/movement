"""Broadcasting operations across ``xarray.DataArray`` dimensions."""

from collections.abc import Callable
from functools import wraps
from typing import Concatenate, ParamSpec, TypeAlias, TypeVar

import numpy as np
from numpy.typing import ArrayLike
from xarray import DataArray

Scalar = TypeVar("Scalar")
Self = TypeVar("Self")
KeywordArgs = ParamSpec("KeywordArgs")
DecoratorInput: TypeAlias = (
    Callable[
        Concatenate[ArrayLike, KeywordArgs],
        Scalar,
    ]
    | Callable[
        Concatenate[Self, ArrayLike, KeywordArgs],
        Scalar,
    ]
)
DecoratorOutput: TypeAlias = Callable[
    Concatenate[DataArray, KeywordArgs], DataArray
]
Decorator: TypeAlias = Callable[[DecoratorInput], DecoratorOutput]


def make_broadcastable(  # noqa: C901
    is_classmethod: bool = False,
    only_broadcastable_along: str | None = None,
) -> Decorator:
    r"""Create a decorator that allows a function to be broadcast.

    Parameters
    ----------
    is_classmethod : bool
        Whether the target of the decoration is a class method which takes
        the ``self`` argument, or a standalone function that receives no
        implicit arguments.
    only_broadcastable_along : str, optional
        Whether the decorated function should only support broadcasting along
        this dimension. The returned function will not take the
        ``broadcast_dimension`` argument, and will use the dimension provided
        here as the value for this argument.

    Returns
    -------
    Decorator
        Decorator function that can be applied with the usual ``@decorator``
        syntax. See Notes for a description of the action of the returned
        decorator.

    Notes
    -----
    The returned decorator (the "``r_decorator``") extends a function that
    acts on a 1D sequence of values, allowing it to be broadcast along the
    axes of an input ``xarray.DataArray``.

    The ``r_decorator`` takes a single parameter, ``f``. ``f`` should be a
    ``Callable`` that acts on 1D data, that is to be converted into a
    broadcast-able function ``fr``, applying the action of ``f`` along an axis
    of an ``xarray.DataArray``.

    If ``f`` is a class method, it should be callable as
    ``f(self, [x, y, ...], \*args, \*\*kwargs)``.
    Otherwise, ``f`` should be callable as
    ``f([x, y, ...], \*args, \*\*kwargs)``.

    The function ``fr`` returned by the ``r_decorator`` is callable with the
    signature
    ``fr([self,] data, \*args, broadcast_dimension = str, \*\*kwargs)``,
    where the ``self`` argument is present only if ``f`` was a class method.
    ``fr`` applies ``f`` along the ``broadcast_dimension`` of ``data``.
    The ``\*args`` and ``\*\*kwargs`` match those passed to ``f``, and retain
    the same interpretations and effects on the result.

    See Also
    --------
    broadcastable_method : Convenience alias for ``is_classmethod = True``.
    space_broadcastable : Convenience alias for
        ``only_broadcastable_along = "space"``.

    Examples
    --------
    Make a standalone function broadcast along the ``"space"`` axis of an
    ``xarray.DataArray``.

    >>> @make_broadcastable(is_classmethod=False, only_broadcast_along="space")
    ... def my_function(xyz_data, *args, **kwargs)
    ...
    ... # Call via the usual arguments, replacing the xyz_data argument with
    ... # the DataArray to broadcast over
    ... my_function(data_array, *args, **kwargs)
    ```

    Make a class method broadcast along any axis of an `xarray.DataArray`.

    >>> from dataclasses import dataclass
    >>> @dataclass
    ... class MyClass:
    ...     factor: float
    ...     offset: float
    ...
    ... @make_broadcastable(is_classmethod=True)
    ... def manipulate_values(self, xyz_values, *args, **kwargs):
    ...     return self.factor * sum(xyz_values) + self.offset
    >>> m = MyClass(factor=5.9, offset=1.0)
    >>> m.manipulate_values(
    ...     data_array, *args, broadcast_dimension="time", **kwargs
    ... )
    ```

    """
    if not only_broadcastable_along:
        only_broadcastable_along = ""

    def make_broadcastable_inner(
        f: DecoratorInput,
    ) -> DecoratorOutput:
        """Broadcast a 1D function along a ``xr.DataArray`` dimension.

        Parameters
        ----------
        f : Callable
            1D function to be converted into a broadcast-able function,, that
            returns a scalar value. If ``f`` is a class method, it should be
            callable as ``f(self, [x, y, ...], *args, **kwargs)``. Otherwise,
            ``f`` should be callable as ``f([x, y, ...], *args, **kwargs).

        Returns
        -------
        Callable
            Callable with signature
            ``(self,) data, *args, broadcast_dimension = str, **kwargs``,
            that applies ``f`` along the ``broadcast_dimension`` of ``data``.
            ``args`` and ``kwargs`` match those passed to ``f``, and retain the
            same interpretations.

        Notes
        -----
        ``mypy`` cannot handle cases where arguments are injected
        into functions: https://github.com/python/mypy/issues/16402.
        As such, we ignore the ``valid-type`` errors where they are flagged by
        the checker in cases such as this ``typing[valid-type]``. Typehints
        provided are consistent with the (expected) input and output types,
        however.

        ``mypy`` does not like when a function, that is to be returned, has its
        signature changed between cases. As such, it recommends defining all
        the possible signatures first, then selecting one using an
        ``if...elif...else`` block. We adhere to this convention in the method
        below.

        """

        @wraps(f)
        def inner_clsmethod(  # type: ignore[valid-type]
            self,
            data: DataArray,
            *args: KeywordArgs.args,
            broadcast_dimension: str = "space",
            **kwargs: KeywordArgs.kwargs,
        ) -> DataArray:
            def f_along_axis(data: ArrayLike, axis: int) -> np.ndarray:
                return np.apply_along_axis(
                    lambda xyz, *a, **kw: f(self, xyz, *a, **kw),
                    axis,
                    data,
                    *args,
                    **kwargs,
                )

            return data.reduce(f_along_axis, broadcast_dimension)

        @wraps(f)
        def inner_clsmethod_fixeddim(
            self,
            data: DataArray,
            *args: KeywordArgs.args,
            **kwargs: KeywordArgs.kwargs,
        ) -> DataArray:
            return inner_clsmethod(
                self,
                data,
                *args,
                broadcast_dimension=only_broadcastable_along,
                **kwargs,
            )

        @wraps(f)
        def inner(  # type: ignore[valid-type]
            data: DataArray,
            *args: KeywordArgs.args,
            broadcast_dimension: str = "space",
            **kwargs: KeywordArgs.kwargs,
        ) -> DataArray:
            def f_along_axis(data: ArrayLike, axis: int) -> np.ndarray:
                return np.apply_along_axis(f, axis, data, *args, **kwargs)

            return data.reduce(f_along_axis, broadcast_dimension)

        @wraps(f)
        def inner_fixeddim(
            data: DataArray,
            *args: KeywordArgs.args,
            **kwargs: KeywordArgs.kwargs,
        ) -> DataArray:
            return inner(
                data,
                *args,
                broadcast_dimension=only_broadcastable_along,
                **kwargs,
            )

        if is_classmethod and only_broadcastable_along:
            return inner_clsmethod_fixeddim
        elif is_classmethod:
            return inner_clsmethod
        elif only_broadcastable_along:
            return inner_fixeddim
        else:
            return inner

    return make_broadcastable_inner


def space_broadcastable(
    is_classmethod: bool = False,
) -> Decorator:
    """Broadcast a 1D function along a ``xr.DataArray`` dimension.

    This is a convenience wrapper for
    ``make_broadcastable(only_broadcastable_along='space')``,
    and is primarily useful when we want to write a function that acts on
    coordinates, that can only be cast across the 'space' dimension of an
    ``xarray.DataArray``.

    See Also
    --------
    make_broadcastable : The aliased decorator function.

    """
    return make_broadcastable(
        is_classmethod=is_classmethod, only_broadcastable_along="space"
    )


def broadcastable_method(
    only_broadcastable_along: str | None = None,
) -> Decorator:
    """Broadcast a class method along a ``xr.DataArray`` dimension.

    This is a convenience wrapper for
    ``make_broadcastable(is_classmethod = True)``,
    for use when extending class methods that act on coordinates, that we wish
    to cast across the axes of an ``xarray.DataArray``.

    See Also
    --------
    make_broadcastable : The aliased decorator function.

    """
    return make_broadcastable(
        is_classmethod=True, only_broadcastable_along=only_broadcastable_along
    )
