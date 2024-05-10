"""Accessors for extending xarray objects."""

import logging
from typing import ClassVar

import xarray as xr

from movement.analysis import kinematics
from movement.io.validators import ValidPosesDataset

logger = logging.getLogger(__name__)

# Preserve the attributes (metadata) of xarray objects after operations
xr.set_options(keep_attrs=True)


@xr.register_dataset_accessor("move")
class MovementDataset:
    """A movement-specicific xarray Dataset accessor.

    Predicted poses over time (pose tracks) are represented in ``movement``
    as and :py:class:`xarray.Dataset` object - a multi-dimensional,
    in-memory array database. Loading functions structure the dataset in
    a ``movement``-specific way, and we refer to such objects as
    'movemen datasets'. A movement dataset is nothing other than a
    :py:class:`xarray.Dataset` with a specific structure and metadata, and
    some added functionality, which is provided in this class.

    You can think of each dataset as a multi-dimensional
    :py:class:`pandas.DataFrame` or as a :py:class:`numpy.ndarray` with
    labelled axes. In ``xarray`` terminology, each axis is called a dimension
    (``dim``), while the labelled "ticks" along each axis are called
    coordinates (``coords``).

    A movement dataset contains the following expected dimensions:
        - ``time``: the number of frames in the video
        - ``individuals``: the number of individuals in the video
        - ``keypoints``: the number of keypoints in the skeleton
        - ``space``: the number of spatial dimensions, either 2 or 3

    Appropriate coordinate labels are assigned to each dimension:
    list of unique names (str) for ``individuals`` and ``keypoints``,
    ['x','y',('z')] for ``space``. The coordinates of the ``time`` dimension
    are in seconds if ``fps`` is provided during loading,
    otherwise they are in frame numbers (increasing integers starting from 0).

    Right after loading a movement dataset, it contains the following
    two data variable stored as :py:class:`xarray.DataArray` objects:
    - ``position``: the 2D or 3D coordinates of the keypoints over time,
    with shape (``time``, ``individuals``, ``keypoints``, ``space``)
    - ``confidence``: the confidence scores of the keypoints over time
    with shape (``time``, ``individuals``, ``keypoints``)

    Grouping data variables together in a single dataset makes it easier to
    keep track of the relationships between them, and makes sense when they
    share some common dimensions (as is the case here).

    Other related data that do not constitute arrays but instead take the form
    of key-value pairs can be stored as attributes - i.e. inside the specially
    designated ``attrs`` dictionary. Right after loading a movement dataset,
    the following attributes are stored:
    - ``fps``: the number of frames per second in the video
    - ``time_unit``: the unit of the ``time`` coordinates, frames or
        seconds
    - ``source_software``: the software from which the poses were loaded
    - ``source_file``: the file from which the poses were
        loaded

    Some of the sample dataset provided with the ``movement`` package
    may also possess additional attributes, such as:
    - ``video_path``: the path to the video file corresponding to the poses
    - ``frame_path``: the path to a single still frame from the video

    All data variables and attributes can be conveniently accessed and
    manipulated using ``xarray``'s built-in interface, for example
    ``ds.position`` or ``ds.fps``.

    Methods/properties that extend the standard xarray functionality are
    are defined in this class and can ve accessed via the ``.move`` accessor,
    e.g. ``ds.move.validate()``.



    Notes
    -----
    Using an accessor is the recommended way to extend xarray objects.
    See [1]_ for more details.


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
            if name.startswith("compute_") and hasattr(kinematics, name):
                self.validate()
                return getattr(kinematics, name)(
                    self._obj.position, *args, **kwargs
                )
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )

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
            error_msg = "The dataset does not contain valid poses."
            logger.error(error_msg)
            raise ValueError(error_msg) from e
