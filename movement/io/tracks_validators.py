import logging
from typing import Any, Iterable, List, Optional, Union

import numpy as np
from attrs import converters, define, field

# get logger
logger = logging.getLogger(__name__)


def _list_of_str(value: Union[str, Iterable[Any]]) -> List[str]:
    """Try to coerce the value into a list of strings.
    Otherwise, raise a ValueError."""
    if type(value) is str:
        warning_msg = (
            f"Invalid value ({value}). Expected a list of strings. "
            "Converting to a list of length 1."
        )
        logger.warning(warning_msg)
        return [value]
    elif isinstance(value, Iterable):
        return [str(item) for item in value]
    else:
        error_msg = f"Invalid value ({value}). Expected a list of strings."
        logger.error(error_msg)
        raise ValueError(error_msg)


def _ensure_type_ndarray(value: Any) -> None:
    """Raise ValueError the value is a not numpy array."""
    if type(value) is not np.ndarray:
        raise ValueError(f"Expected a numpy array, but got {type(value)}.")


def _set_fps_to_none_if_invalid(fps: Optional[float]) -> Optional[float]:
    """Set fps to None if a non-positive float is passed."""
    if fps is not None and fps <= 0:
        logger.warning(
            f"Invalid fps value ({fps}). Expected a positive number. "
            "Setting fps to None."
        )
        return None
    return fps


@define(kw_only=True)
class ValidPoseTracks:
    """Class for validating pose tracking data imported from files, before
    they are converted to a `PoseTracks` object.

    Attributes
    ----------
    tracks_array : np.ndarray
        Array of shape (n_frames, n_individuals, n_keypoints, n_space)
        containing the pose tracks. It will be converted to a
        `xarray.DataArray` object named "pose_tracks".
    scores_array : np.ndarray, optional
        Array of shape (n_frames, n_individuals, n_keypoints) containing
        the point-wise confidence scores. It will be converted to a
        `xarray.DataArray` object named "confidence".
        If None (default), the scores will be set to an array of NaNs.
    individual_names : list of str, optional
        List of unique names for the individuals in the video. If None
        (default), the individuals will be named "individual_0",
        "individual_1", etc.
    keypoint_names : list of str, optional
        List of unique names for the keypoints in the skeleton. If None
        (default), the keypoints will be named "keypoint_0", "keypoint_1",
        etc.
    fps : float, optional
        Frames per second of the video. Defaults to None.
    """

    # Define class attributes
    tracks_array: np.ndarray = field()
    scores_array: Optional[np.ndarray] = field(default=None)
    individual_names: Optional[List[str]] = field(
        default=None,
        converter=converters.optional(_list_of_str),
    )
    keypoint_names: Optional[List[str]] = field(
        default=None,
        converter=converters.optional(_list_of_str),
    )
    fps: Optional[float] = field(
        default=None,
        converter=converters.pipe(  # type: ignore
            converters.optional(float), _set_fps_to_none_if_invalid
        ),
    )

    # Add validators
    @tracks_array.validator
    def _validate_tracks_array(self, attribute, value):
        _ensure_type_ndarray(value)
        if value.ndim != 4:
            raise ValueError(
                f"Expected `{attribute}` to have 4 dimensions, "
                f"but got {value.ndim}."
            )
        if value.shape[-1] not in [2, 3]:
            raise ValueError(
                f"Expected `{attribute}` to have 2 or 3 spatial dimensions, "
                f"but got {value.shape[-1]}."
            )

    @scores_array.validator
    def _validate_scores_array(self, attribute, value):
        if value is not None:
            _ensure_type_ndarray(value)
            if value.shape != self.tracks_array.shape[:-1]:
                raise ValueError(
                    f"Expected `{attribute}` to have shape "
                    f"{self.tracks_array.shape[:-1]}, but got {value.shape}."
                )

    @individual_names.validator
    def _validate_individual_names(self, attribute, value):
        if (value is not None) and (len(value) != self.tracks_array.shape[1]):
            raise ValueError(
                f"Expected {self.tracks_array.shape[1]} `{attribute}`, "
                f"but got {len(value)}."
            )

    @keypoint_names.validator
    def _validate_keypoint_names(self, attribute, value):
        if (value is not None) and (len(value) != self.tracks_array.shape[2]):
            raise ValueError(
                f"Expected {self.tracks_array.shape[2]} `{attribute}`, "
                f"but got {len(value)}."
            )

    def __attrs_post_init__(self):
        """Assign default values to optional attributes (if None)"""
        if self.scores_array is None:
            self.scores_array = np.full(
                (self.tracks_array.shape[:-1]), np.nan, dtype="float32"
            )
            logger.warning(
                "Scores array was not provided. Setting to an array of NaNs."
            )
        if self.individual_names is None:
            self.individual_names = [
                f"individual_{i}" for i in range(self.tracks_array.shape[1])
            ]
            logger.warning(
                "Individual names were not provided. "
                f"Setting to {self.individual_names}."
            )
        if self.keypoint_names is None:
            self.keypoint_names = [
                f"keypoint_{i}" for i in range(self.tracks_array.shape[2])
            ]
            logger.warning(
                "Keypoint names were not provided. "
                f"Setting to {self.keypoint_names}."
            )
