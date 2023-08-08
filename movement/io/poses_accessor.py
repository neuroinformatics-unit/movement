import logging
from typing import ClassVar

import xarray as xr

from movement.io.validators import ValidPoseTracks

logger = logging.getLogger(__name__)

# Preserve the attributes (metadata) of xarray objects after operations
xr.set_options(keep_attrs=True)


@xr.register_dataset_accessor("poses")
class PosesAccessor:
    """An accessor that extends an `xarray.Dataset` object.

    The `xarray.Dataset` has the following dimensions:
    - `time`: the number of frames in the video
    - `individuals`: the number of individuals in the video
    - `keypoints`: the number of keypoints in the skeleton
    - `space`: the number of spatial dimensions, either 2 or 3

    Appropriate coordinate labels are assigned to each dimension:
    list of unique names (str) for `individuals` and `keypoints`,
    ['x','y',('z')] for `space`. The coordinates of the `time` dimension are
    in seconds if `fps` is provided, otherwise they are in frame numbers.

    The dataset contains two data variables (`xarray.DataArray` objects):
    - `pose_tracks`: with shape (`time`, `individuals`, `keypoints`, `space`)
    - `confidence`: with shape (`time`, `individuals`, `keypoints`)

    The dataset may also contain following attributes as metadata:
    - `fps`: the number of frames per second in the video
    - `time_unit`: the unit of the `time` coordinates, frames or seconds
    - `source_software`: the software from which the pose tracks were loaded
    - `source_file`: the file from which the pose tracks were loaded

    Notes
    -----
    Using an acessor is the recommended way to extend xarray objects.
    See [1]_ for more details.

    Methods/properties that are specific to this class can be used via
    the `.poses` accessor, e.g. `ds.poses.to_dlc_df()`.

    References
    ----------
    .. _1: https://docs.xarray.dev/en/stable/internals/extending-xarray.html
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

    def validate(self) -> None:
        """Validate the PoseTracks dataset."""
        fps = self._obj.attrs.get("fps", None)
        try:
            ValidPoseTracks(
                tracks_array=self._obj[self.var_names[0]].values,
                scores_array=self._obj[self.var_names[1]].values,
                individual_names=self._obj.coords[self.dim_names[1]].values,
                keypoint_names=self._obj.coords[self.dim_names[2]].values,
                fps=fps,
            )
        except Exception as e:
            error_msg = "The dataset does not contain valid pose tracks."
            logger.error(error_msg)
            raise ValueError(error_msg) from e
