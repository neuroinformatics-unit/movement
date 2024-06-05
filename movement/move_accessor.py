"""Accessor for extending xarray.Dataset objects."""

import logging
from typing import ClassVar

import xarray as xr

from movement.analysis import kinematics
from movement.io.validators import ValidPosesDataset
from movement.logging import log_error

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


    References
    ----------
    .. [1] https://docs.xarray.dev/en/stable/internals/extending-xarray.html

    """

    # Names of the expected dimensions in the dataset
    dim_names: ClassVar[tuple] = (
        "time",
        "individuals",
        "keypoints",
        "space",
    )

    # Names of the expected data variables in the dataset
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
        to the respective functions in the ``kinematics``  module.

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
            if not name.startswith("compute_") or not hasattr(
                kinematics, name
            ):
                error_msg = (
                    f"'{self.__class__.__name__}' object has "
                    f"no attribute '{name}'"
                )
                raise log_error(AttributeError, error_msg)
            if not hasattr(self._obj, "position"):
                raise log_error(
                    AttributeError,
                    "Missing required data variables: 'position'",
                )
            try:
                return getattr(kinematics, name)(
                    self._obj.position, *args, **kwargs
                )
            except Exception as e:
                error_msg = f"Failed to evoke '{name}'. "
                raise log_error(AttributeError, error_msg) from e

        return method

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
