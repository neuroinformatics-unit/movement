r"""Broadcasting operations across ``xarray.DataArray`` dimensions.

This module essentially provides an equivalent functionality to
``numpy.apply_along_axis``, but for ``xarray.DataArray`` objects.
This functionality is provided as a decorator, so it can be applied to both
functions within the package and be available to users who would like to use it
in their analysis.
In essence; suppose that we have a function which takes a 1D-slice of a
``xarray.DataArray`` and returns either a scalar value, or another 1D array.
Typically, one would either have to call this function successively in a
``for`` loop, looping over all the 1D slices in a ``xarray.DataArray`` that
need to be examined, or re-write the function to be able to broadcast along the
necessary dimension of the data structure.

The ``make_broadcastable`` decorator takes care of the latter piece of work,
allowing us to write functions that operate on 1D slices, then apply this
decorator to have them work across ``xarray.DataArray`` dimensions. The
function

>>> def my_function(input_1d, *args, **kwargs):
...     # do something
...     return scalar_or_1d_output

which previously only worked with 1D-slices can be decorated

>>> @make_broadcastable()
... def my_function(input_1d, *args, **kwargs):
...     # do something
...     return scalar_or_1d_output

effectively changing its call signature to

>>> def my_function(data_array, *args, dimension, **kwargs):
...     # do my_function, but do it to all the slices
...     # along the dimension of data_array.
...     return data_array_output

which will perform the action of ``my_function`` along the ``dimension`` given.
The ``*args`` and ``**kwargs`` retain their original interpretations from
``my_function`` too.
"""

from collections.abc import Callable
from functools import wraps
from typing import Concatenate, ParamSpec, TypeAlias, TypeVar

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike

ScalarOr1D = TypeVar("ScalarOr1D", float, int, bool, ArrayLike)
Self = TypeVar("Self")
KeywordArgs = ParamSpec("KeywordArgs")
ClsMethod1DTo1D = Callable[
    Concatenate[Self, ArrayLike, KeywordArgs],
    ScalarOr1D,
]
Function1DTo1D: TypeAlias = Callable[
    Concatenate[ArrayLike, KeywordArgs],
    ScalarOr1D,
]
FunctionDaToDa: TypeAlias = Callable[
    Concatenate[xr.DataArray, KeywordArgs], xr.DataArray
]
DecoratorInput: TypeAlias = Function1DTo1D | ClsMethod1DTo1D
Decorator: TypeAlias = Callable[[DecoratorInput], FunctionDaToDa]


def apply_along_da_axis(
    f: Callable[[ArrayLike], ScalarOr1D],
    data: xr.DataArray,
    dimension: str,
    new_dimension_name: str | None = None,
) -> xr.DataArray:
    """Apply a function ``f`` across ``dimension`` of ``data``.

    ``f`` should be callable as ``f(input_1D)`` where ``input_1D`` is a one-
    dimensional ``numpy.typing.ArrayLike`` object. It should return either a
    scalar or one-dimensional ``numpy.typing.ArrayLike`` object.

    Parameters
    ----------
    f : Callable
        Function that takes 1D inputs and returns either scalar or 1D outputs.
        This will be cast across the ``dimension`` of the ``data``.
    data: xarray.DataArray
        Values to be cast over.
    dimension : str
        Dimension of ``data`` to broadcast ``f`` across.
    new_dimension_name : str, optional
        If ``f`` returns non-scalar values, the dimension in the output that
        these values are returned along is given the name
        ``new_dimension_name``. Defaults to ``"result"``.

    Returns
    -------
    xarray.DataArray
        Result of broadcasting ``f`` along the ``dimension`` of ``data``.

        - If ``f`` returns a scalar or ``(1,)``-shaped output, the output has
          one fewer dimension than ``data``, with ``dimension`` being dropped.
          All other dimensions retain their names and sizes.
        - If ``f`` returns a ``(n,)``-shaped output for ``n > 1``; all non-
          ``dimension`` dimensions of ``data`` retain their shapes. The
          ``dimension`` dimension itself is replaced with a new dimension,
          ``new_dimension_name``, containing the output of the application of
          ``f``.

    """
    output: xr.DataArray = xr.apply_ufunc(
        lambda input_1D: np.atleast_1d(f(input_1D)),
        data,
        input_core_dims=[[dimension]],
        exclude_dims=set((dimension,)),
        output_core_dims=[[dimension]],
        vectorize=True,
    )
    if len(output[dimension]) < 2:
        output = output.squeeze(dim=dimension)
    else:
        # Rename the non-1D output dimension according to request
        output = output.rename(
            {dimension: new_dimension_name if new_dimension_name else "result"}
        )
    return output


def make_broadcastable(  # noqa: C901
    is_classmethod: bool = False,
    only_broadcastable_along: str | None = None,
    new_dimension_name: str | None = None,
) -> Decorator:
    """Create a decorator that allows a function to be broadcast.

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
    new_dimension_name : str, optional
        Passed to :func:`apply_along_da_axis`.

    Returns
    -------
    Decorator
        Decorator function that can be applied with the
        ``@make_broadcastable(...)`` syntax. See Notes for a description of
        the action of the returned decorator.

    Notes
    -----
    The returned decorator (the "``r_decorator``") extends a function that
    acts on a 1D sequence of values, allowing it to be broadcast along the
    axes of an input ``xarray.DataArray``.

    The ``r_decorator`` takes a single parameter, ``f``. ``f`` should be a
    ``Callable`` that acts on 1D inputs, that is to be converted into a
    broadcast-able function ``fr``, applying the action of ``f`` along an axis
    of an ``xarray.DataArray``. ``f`` should return either scalar or 1D
    outputs.

    If ``f`` is a class method, it should be callable as
    ``f(self, [x, y, ...], *args, **kwargs)``.
    Otherwise, ``f`` should be callable as
    ``f([x, y, ...], *args, **kwargs)``.

    The function ``fr`` returned by the ``r_decorator`` is callable with the
    signature
    ``fr([self,] data, *args, broadcast_dimension = str, **kwargs)``,
    where the ``self`` argument is present only if ``f`` was a class method.
    ``fr`` applies ``f`` along the ``broadcast_dimension`` of ``data``.
    The ``*args`` and ``**kwargs`` match those passed to ``f``, and retain
    the same interpretations and effects on the result. If ``data`` provided to
    ``fr`` is not an ``xarray.DataArray``, it will fall back on the behaviour
    of ``f`` (and ignore the ``broadcast_dimension`` argument).

    See the docstring of ``make_broadcastable_inner`` in the source code for a
    more explicit explanation of the returned decorator.

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
    ... # the xarray.DataArray to broadcast over
    ... my_function(data_array, *args, **kwargs)
    ```

    Make a class method broadcast along any axis of an `xarray.DataArray`.

    >>> from dataclasses import dataclass
    >>>
    >>> @dataclass
    ... class MyClass:
    ...     factor: float
    ...     offset: float
    ...
    ...     @make_broadcastable(is_classmethod=True)
    ...     def manipulate_values(self, xyz_values, *args, **kwargs):
    ...         return self.factor * sum(xyz_values) + self.offset
    ...
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
    ) -> FunctionDaToDa:
        """Broadcast a 1D function along a ``xarray.DataArray`` dimension.

        Parameters
        ----------
        f : Callable
            1D function to be converted into a broadcast-able function,, that
            returns either a scalar value or 1D output. If ``f`` is a class
            method, it should be callable as
            ``f(self, [x, y, ...], *args, **kwargs)``.
            Otherwise, ``f`` should be callable as
            ``f([x, y, ...], *args, **kwargs).

        Returns
        -------
        Callable
            Callable with signature
            ``(self,) data, *args, broadcast_dimension = str, **kwargs``,
            that applies ``f`` along the ``broadcast_dimension`` of ``data``.
            ``*args`` and ``**kwargs`` match those passed to ``f``, and
            retain the same interpretations.

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
        def inner_clsmethod(
            self,
            data: xr.DataArray,
            *args: KeywordArgs.args,  # type: ignore[valid-type]
            broadcast_dimension: str = "space",
            **kwargs: KeywordArgs.kwargs,  # type: ignore[valid-type]
        ) -> xr.DataArray:
            # Preserve original functionality
            if not isinstance(data, xr.DataArray):
                return f(self, data, *args, **kwargs)
            return apply_along_da_axis(
                lambda input_1D: f(self, input_1D, *args, **kwargs),
                data,
                broadcast_dimension,
                new_dimension_name=new_dimension_name,
            )

        @wraps(f)
        def inner_clsmethod_fixeddim(
            self,
            data: xr.DataArray,
            *args: KeywordArgs.args,  # type: ignore[valid-type]
            **kwargs: KeywordArgs.kwargs,  # type: ignore[valid-type]
        ) -> xr.DataArray:
            return inner_clsmethod(
                self,
                data,
                *args,
                broadcast_dimension=only_broadcastable_along,
                **kwargs,
            )

        @wraps(f)
        def inner(
            data: xr.DataArray,
            *args: KeywordArgs.args,  # type: ignore[valid-type]
            broadcast_dimension: str = "space",
            **kwargs: KeywordArgs.kwargs,  # type: ignore[valid-type]
        ) -> xr.DataArray:
            # Preserve original functionality
            if not isinstance(data, xr.DataArray):
                return f(data, *args, **kwargs)
            return apply_along_da_axis(
                lambda input_1D: f(input_1D, *args, **kwargs),
                data,
                broadcast_dimension,
                new_dimension_name=new_dimension_name,
            )

        @wraps(f)
        def inner_fixeddim(
            data: xr.DataArray,
            *args: KeywordArgs.args,  # type: ignore[valid-type]
            **kwargs: KeywordArgs.kwargs,  # type: ignore[valid-type]
        ) -> xr.DataArray:
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
    new_dimension_name: str | None = None,
) -> Decorator:
    """Broadcast a 1D function along the 'space' dimension.

    This is a convenience wrapper for
    ``make_broadcastable(only_broadcastable_along='space')``,
    and is primarily useful when we want to write a function that acts on
    coordinates, that can only be cast across the 'space' dimension of an
    ``xarray.DataArray``.

    Returns
    -------
    Callable
        Callable with signature
        ``(self,) data, *args, broadcast_dimension = str, **kwargs``,
        that applies ``f`` along the ``broadcast_dimension`` of ``data``.
        ``*args`` and ``**kwargs`` match those passed to ``f``, and
        retain the same interpretations.

    See Also
    --------
    make_broadcastable : The aliased decorator function.

    """
    return make_broadcastable(
        is_classmethod=is_classmethod,
        only_broadcastable_along="space",
        new_dimension_name=new_dimension_name,
    )


def broadcastable_method(
    only_broadcastable_along: str | None = None,
    new_dimension_name: str | None = None,
) -> Decorator:
    """Broadcast a class method along a ``xarray.DataArray`` dimension.

    This is a convenience wrapper for
    ``make_broadcastable(is_classmethod = True)``,
    for use when extending class methods that act on coordinates, that we wish
    to cast across the axes of an ``xarray.DataArray``.

    Returns
    -------
    Callable
        Callable with signature
        ``(self,) data, *args, broadcast_dimension = str, **kwargs``,
        that applies ``f`` along the ``broadcast_dimension`` of ``data``.
        ``*args`` and ``**kwargs`` match those passed to ``f``, and
        retain the same interpretations.

    See Also
    --------
    make_broadcastable : The aliased decorator function.

    """
    return make_broadcastable(
        is_classmethod=True,
        only_broadcastable_along=only_broadcastable_along,
        new_dimension_name=new_dimension_name,
    )
