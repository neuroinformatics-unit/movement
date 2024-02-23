import logging
from typing import Callable, ClassVar

import xarray as xr

from movement.analysis import kinematics
from movement.io.validators import ValidPoseTracks

logger = logging.getLogger(__name__)

# Preserve the attributes (metadata) of xarray objects after operations
xr.set_options(keep_attrs=True)


@xr.register_dataset_accessor("move")
class MoveAccessor:
    """An accessor that extends an xarray Dataset by implementing
    `movement`-specific properties and methods.

    The xarray Dataset contains the following expected dimensions:
        - ``time``: the number of frames in the video
        - ``individuals``: the number of individuals in the video
        - ``keypoints``: the number of keypoints in the skeleton
        - ``space``: the number of spatial dimensions, either 2 or 3

    Appropriate coordinate labels are assigned to each dimension:
    list of unique names (str) for ``individuals`` and ``keypoints``,
    ['x','y',('z')] for ``space``. The coordinates of the ``time`` dimension
    are in seconds if ``fps`` is provided, otherwise they are in frame numbers.

    The dataset contains two expected data variables (xarray DataArrays):
        - ``pose_tracks``: with shape (``time``, ``individuals``,
          ``keypoints``, ``space``)
        - ``confidence``: with shape (``time``, ``individuals``, ``keypoints``)

    When accessing a ``.move`` property (e.g. ``displacement``, ``velocity``,
    ``acceleration``) for the first time, the property is computed and stored
    as a data variable with the same name in the dataset. The ``.move``
    accessor can be omitted in subsequent accesses, i.e.
    ``ds.move.displacement`` and ``ds.displacement`` will return the same data
    variable.

    The dataset may also contain following attributes as metadata:
        - ``fps``: the number of frames per second in the video
        - ``time_unit``: the unit of the ``time`` coordinates, frames or
          seconds
        - ``source_software``: the software from which the pose tracks were
          loaded
        - ``source_file``: the file from which the pose tracks were
          loaded

    Notes
    -----
    Using an accessor is the recommended way to extend xarray objects.
    See [1]_ for more details.

    Methods/properties that are specific to this class can be accessed via
    the ``.move`` accessor, e.g. ``ds.move.validate()``.

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
        "pose_tracks",
        "confidence",
    )

    def __init__(self, ds: xr.Dataset):
        self._obj = ds

    def _compute_property(
        self,
        property: str,
        compute_function: Callable[[xr.DataArray], xr.DataArray],
    ) -> xr.DataArray:
        """Compute a kinematic property and store it in the dataset.

        Parameters
        ----------
        property : str
            The name of the property to compute.
        compute_function : Callable[[xarray.DataArray], xarray.DataArray]
            The function to compute the property.

        Returns
        -------
        xarray.DataArray
            The computed property.
        """
        self.validate()
        if property not in self._obj:
            pose_tracks = self._obj[self.var_names[0]]
            self._obj[property] = compute_function(pose_tracks)
        return self._obj[property]

    @property
    def displacement(self) -> xr.DataArray:
        """Return the displacement between consecutive positions
        of each keypoint for each individual across time.
        """
        return self._compute_property(
            "displacement", kinematics.compute_displacement
        )

    @property
    def velocity(self) -> xr.DataArray:
        """Return the velocity between consecutive positions
        of each keypoint for each individual across time.
        """
        return self._compute_property("velocity", kinematics.compute_velocity)

    @property
    def acceleration(self) -> xr.DataArray:
        """Return the acceleration between consecutive positions
        of each keypoint for each individual across time.
        """
        return self._compute_property(
            "acceleration", kinematics.compute_acceleration
        )

    def validate(self) -> None:
        """Validate the PoseTracks dataset."""
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
            ValidPoseTracks(
                tracks_array=self._obj[self.var_names[0]].values,
                scores_array=self._obj[self.var_names[1]].values,
                individual_names=self._obj.coords[self.dim_names[1]].values,
                keypoint_names=self._obj.coords[self.dim_names[2]].values,
                fps=fps,
                source_software=source_software,
            )
        except Exception as e:
            error_msg = "The dataset does not contain valid pose tracks."
            logger.error(error_msg)
            raise ValueError(error_msg) from e
