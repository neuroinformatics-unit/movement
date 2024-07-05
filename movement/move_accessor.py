"""Accessor for extending :py:class:`xarray.Dataset` objects."""

import logging
from typing import ClassVar

import xarray as xr

from movement import filtering
from movement.analysis import kinematics
from movement.utils.logging import log_error
from movement.validators.datasets import ValidPosesDataset

logger = logging.getLogger(__name__)

# Preserve the attributes (metadata) of xarray objects after operations
xr.set_options(keep_attrs=True)


@xr.register_dataset_accessor("move")
class MovementDataset:
    """An :py:class:`xarray.Dataset` accessor for pose tracking data.

    A ``movement`` dataset is an :py:class:`xarray.Dataset` with a specific
    structure to represent pose tracks, associated confidence scores and
    relevant metadata.

    Methods/properties that extend the standard ``xarray`` functionality are
    defined in this class. To avoid conflicts with ``xarray``'s namespace,
    ``movement``-specific methods are accessed using the ``move`` keyword,
    for example ``ds.move.validate()`` (see [1]_ for more details).

    Attributes
    ----------
    dim_names : tuple
        Names of the expected dimensions in the dataset.
    var_names : tuple
        Names of the expected data variables in the dataset.

    References
    ----------
    .. [1] https://docs.xarray.dev/en/stable/internals/extending-xarray.html

    """

    dim_names: ClassVar[tuple] = (
        "time",
        "individuals",
        "keypoints",
        "space",
    )

    var_names: ClassVar[tuple] = (
        "position",
        "confidence",
    )

    def __init__(self, ds: xr.Dataset):
        """Initialize the MovementDataset."""
        self._obj = ds

    def __getattr__(self, name: str) -> xr.DataArray:
        """Forward requested but undefined attributes to relevant modules.

        This method currently only forwards kinematic property computation
        and filtering operations to the respective functions in
        :py:mod:`movement.analysis.kinematics` and
        :py:mod:`movement.filtering`.

        Parameters
        ----------
        name : str
            The name of the attribute to get.

        Returns
        -------
        xarray.DataArray
            The computed attribute value.

        Raises
        ------
        AttributeError
            If the attribute does not exist.

        """

        def method(*args, **kwargs):
            if hasattr(kinematics, name):
                return self.kinematics_wrapper(name, *args, **kwargs)
            elif hasattr(filtering, name):
                return self.filtering_wrapper(name, *args, **kwargs)
            else:
                error_msg = (
                    f"'{self.__class__.__name__}' object has "
                    f"no attribute '{name}'"
                )
                raise log_error(AttributeError, error_msg)

        return method

    def kinematics_wrapper(
        self, fn_name: str, *args, **kwargs
    ) -> xr.DataArray:
        """Provide convenience method for computing kinematic properties.

        This method forwards kinematic property computation
        to the respective functions in :py:mod:`movement.analysis.kinematics`.

        Parameters
        ----------
        fn_name : str
            The name of the kinematics function to call.
        args : tuple
            Positional arguments to pass to the function.
        kwargs : dict
            Keyword arguments to pass to the function.

        Returns
        -------
        xarray.DataArray
            The computed kinematics attribute value.

        Raises
        ------
        RuntimeError
            If the requested function fails to execute.

        Examples
        --------
        Compute ``displacement`` based on the ``position`` data variable
        in the Dataset ``ds`` and store the result in ``ds``.

        >>> ds["displacement"] = ds.move.compute_displacement()

        Compute ``velocity`` based on the ``position`` data variable in
        the Dataset ``ds`` and store the result in ``ds``.

        >>> ds["velocity"] = ds.move.compute_velocity()

        Compute ``acceleration`` based on the ``position`` data variable
        in the Dataset ``ds`` and store the result in ``ds``.

        >>> ds["acceleration"] = ds.move.compute_acceleration()

        """
        try:
            return getattr(kinematics, fn_name)(
                self._obj.position, *args, **kwargs
            )
        except Exception as e:
            error_msg = (
                f"Failed to evoke '{fn_name}' via 'move' accessor. {str(e)}"
            )
            raise log_error(RuntimeError, error_msg) from e

    def filtering_wrapper(
        self, fn_name: str, *args, data_vars: list[str] | None = None, **kwargs
    ) -> xr.DataArray | dict[str, xr.DataArray]:
        """Provide convenience method for filtering data variables.

        This method forwards filtering and/or smoothing to the respective
        functions in :py:mod:`movement.filtering`. The data variables to
        filter can be specified in ``data_vars``. If ``data_vars`` is not
        specified, the ``position`` data variable is selected by default.

        Parameters
        ----------
        fn_name : str
            The name of the filtering function to call.
        args : tuple
            Positional arguments to pass to the function.
        data_vars : list[str] | None
            The data variables to apply filtering. If ``None``, the
            ``position`` data variable will be passed by default.
        kwargs : dict
            Keyword arguments to pass to the function.

        Returns
        -------
        xarray.DataArray | dict[str, xarray.DataArray]
            The filtered data variable or a dictionary of filtered data
            variables, if multiple data variables are specified.

        Raises
        ------
        RuntimeError
            If the requested function fails to execute.

        Examples
        --------
        Filter the ``position`` data variable to drop points with
        ``confidence`` below 0.7 and store the result back into the
        Dataset ``ds``.
        Since ``data_vars`` is not supplied, the filter will be applied to
        the ``position`` data variable by default.

        >>> ds["position"] = ds.move.filter_by_confidence(threshold=0.7)

        Apply a median filter to the ``position`` data variable and
        store this back into the Dataset ``ds``.

        >>> ds["position"] = ds.move.median_filter(window=3)

        Apply a Savitzky-Golay filter to both the ``position`` and
        ``velocity`` data variables and store these back into the
        Dataset ``ds``. ``filtered_data`` is a dictionary, where the keys
        are the data variable names and the values are the filtered
        DataArrays.

        >>> filtered_data = ds.move.savgol_filter(
        ...     window=3, data_vars=["position", "velocity"]
        ... )
        >>> ds.update(filtered_data)

        """
        ds = self._obj
        if data_vars is None:  # Default to filter on position
            data_vars = ["position"]
        if fn_name == "filter_by_confidence":
            # Add confidence to kwargs
            kwargs["confidence"] = ds.confidence
        try:
            result = {
                data_var: getattr(filtering, fn_name)(
                    ds[data_var], *args, **kwargs
                )
                for data_var in data_vars
            }
            # Return DataArray if result only has one key
            if len(result) == 1:
                return result[list(result.keys())[0]]
            return result
        except Exception as e:
            error_msg = (
                f"Failed to evoke '{fn_name}' via 'move' accessor. {str(e)}"
            )
            raise log_error(RuntimeError, error_msg) from e

    def validate(self) -> None:
        """Validate the dataset.

        This method checks if the dataset contains the expected dimensions,
        data variables, and metadata attributes. It also ensures that the
        dataset contains valid poses.
        """
        fps = self._obj.attrs.get("fps", None)
        source_software = self._obj.attrs.get("source_software", None)
        try:
            missing_dims = set(self.dim_names) - set(self._obj.dims)
            missing_vars = set(self.var_names) - set(self._obj.data_vars)
            if missing_dims:
                raise ValueError(
                    f"Missing required dimensions: {missing_dims}"
                )
            if missing_vars:
                raise ValueError(
                    f"Missing required data variables: {missing_vars}"
                )
            ValidPosesDataset(
                position_array=self._obj[self.var_names[0]].values,
                confidence_array=self._obj[self.var_names[1]].values,
                individual_names=self._obj.coords[self.dim_names[1]].values,
                keypoint_names=self._obj.coords[self.dim_names[2]].values,
                fps=fps,
                source_software=source_software,
            )
        except Exception as e:
            error_msg = "The dataset does not contain valid poses. " + str(e)
            raise log_error(ValueError, error_msg) from e
