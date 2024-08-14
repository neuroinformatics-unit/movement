"""Accessor for extending :class:`xarray.Dataset` objects."""

import logging
from typing import ClassVar

import xarray as xr

from movement import filtering
from movement.analysis import kinematics
from movement.utils.logging import log_error
from movement.validators.datasets import ValidBboxesDataset, ValidPosesDataset

logger = logging.getLogger(__name__)

# Preserve the attributes (metadata) of xarray objects after operations
xr.set_options(keep_attrs=True)


@xr.register_dataset_accessor("move")
class MovementDataset:
    """An :class:`xarray.Dataset` accessor for ``movement`` data.

    A ``movement`` dataset is an :class:`xarray.Dataset` with a specific
    structure to represent pose tracks or bounding boxes data,
    associated confidence scores and relevant metadata.

    Methods/properties that extend the standard ``xarray`` functionality are
    defined in this class. To avoid conflicts with ``xarray``'s namespace,
    ``movement``-specific methods are accessed using the ``move`` keyword,
    for example ``ds.move.validate()`` (see [1]_ for more details).

    Attributes
    ----------
    dim_names : dict
        A dictionary with the names of the expected dimensions in the dataset,
        for each dataset type (``"poses"`` or ``"bboxes"``).
    var_names : dict
        A dictionary with the expected data variables in the dataset, for each
        dataset type (``"poses"`` or ``"bboxes"``).

    References
    ----------
    .. [1] https://docs.xarray.dev/en/stable/internals/extending-xarray.html

    """

    # Set class attributes for expected dimensions and data variables
    dim_names: ClassVar[dict] = {
        "poses": ("time", "individuals", "keypoints", "space"),
        "bboxes": ("time", "individuals", "space"),
    }
    var_names: ClassVar[dict] = {
        "poses": ("position", "confidence"),
        "bboxes": ("position", "shape", "confidence"),
    }

    def __init__(self, ds: xr.Dataset):
        """Initialize the MovementDataset."""
        self._obj = ds
        # Set instance attributes based on dataset type
        self.dim_names_instance = self.dim_names[self._obj.ds_type]
        self.var_names_instance = self.var_names[self._obj.ds_type]

    def __getattr__(self, name: str) -> xr.DataArray:
        """Forward requested but undefined attributes to relevant modules.

        This method currently only forwards kinematic property computation
        and filtering operations to the respective functions in
        :mod:`movement.analysis.kinematics` and
        :mod:`movement.filtering`.

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
        to the respective functions in :mod:`movement.analysis.kinematics`.

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
        functions in :mod:`movement.filtering`. The data variables to
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
        dataset contains valid poses or bounding boxes data.

        Raises
        ------
        ValueError
            If the dataset is missing required dimensions, data variables,
            or contains invalid poses or bounding boxes data.

        """
        fps = self._obj.attrs.get("fps", None)
        source_software = self._obj.attrs.get("source_software", None)
        try:
            self._validate_dims()
            self._validate_data_vars()
            if self._obj.ds_type == "poses":
                ValidPosesDataset(
                    position_array=self._obj["position"].values,
                    confidence_array=self._obj["confidence"].values,
                    individual_names=self._obj.coords["individuals"].values,
                    keypoint_names=self._obj.coords["keypoints"].values,
                    fps=fps,
                    source_software=source_software,
                )
            elif self._obj.ds_type == "bboxes":
                # Define frame_array.
                # Recover from time axis in seconds if necessary.
                frame_array = self._obj.coords["time"].values.reshape(-1, 1)
                if self._obj.attrs["time_unit"] == "seconds":
                    frame_array *= fps
                ValidBboxesDataset(
                    position_array=self._obj["position"].values,
                    shape_array=self._obj["shape"].values,
                    confidence_array=self._obj["confidence"].values,
                    individual_names=self._obj.coords["individuals"].values,
                    frame_array=frame_array,
                    fps=fps,
                    source_software=source_software,
                )
        except Exception as e:
            error_msg = (
                f"The dataset does not contain valid {self._obj.ds_type}. {e}"
            )
            raise log_error(ValueError, error_msg) from e

    def _validate_dims(self) -> None:
        missing_dims = set(self.dim_names_instance) - set(self._obj.dims)
        if missing_dims:
            raise ValueError(
                f"Missing required dimensions: {sorted(missing_dims)}"
            )  # sort for a reproducible error message

    def _validate_data_vars(self) -> None:
        missing_vars = set(self.var_names_instance) - set(self._obj.data_vars)
        if missing_vars:
            raise ValueError(
                f"Missing required data variables: {sorted(missing_vars)}"
            )  # sort for a reproducible error message
