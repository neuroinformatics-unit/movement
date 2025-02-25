import numpy as np
import xarray as xr
from collections.abc import Hashable


def np_array_to_compatible_dataarray(
    array: np.ndarray,
    target_da: xr.DataArray,
    essential_dimensions: list[Hashable],
) -> None:
    """"""
    nontrival_array_dims = array.squeeze().shape
    nontrival_da = target_da.squeeze(drop=True)
    nontrival_da_dims = target_da.dims

    # All non-trivial dimensions in array must appear in the target_da.
    # They might not necessarily be in order.
    # Two dimensions cannot match to one in the target DA, so using set is out the window as the number of occurances of a value also matters.
    for dim_size in nontrival_array_dims.unique():
        n_dims_this_size = (nontrival_array_dims == dim_size).sum()
        n_dims_this_size_in_target = ...
        assert n_dims_this_size <= n_dims_this_size_in_target, (
            "otherwise can't continue"
        )
    else:
        # All non-trivial dimensions in the array are matched to a dimension of the same length in the target_da. Assign them the dimension labels, by default we go in order in the event of ties.
        dim_order = sorted(
            range(len(nontrival_array_dims)),
            key=lambda x: nontrival_array_dims[x],
        )
        dim_names = sorted(
            nontrival_da_dims, key=lambda x: nontrival_da.shape[x]
        )
        dim_names = sorted(
            dim_names, key=lambda x: dim_order[dim_names.index(x)]
        )
    return


target = xr.DataArray(np.zeros((2, 3, 4)), dims=["a", "b", "c"])
no_matching_dims = np.zeros((1, 5))
matches_a = np.zeros((5, 2))

np_array_to_compatible_dataarray()
