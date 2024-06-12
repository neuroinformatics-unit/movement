"""`attrs` classes for validating data structures."""

from collections.abc import Iterable
from typing import Any, Optional, Union

import numpy as np
from attrs import converters, define, field, validators

from movement.logging import log_error, log_warning


def _list_of_str(value: Union[str, Iterable[Any]]) -> list[str]:
    """Try to coerce the value into a list of strings."""
    if isinstance(value, str):
        log_warning(
            f"Invalid value ({value}). Expected a list of strings. "
            "Converting to a list of length 1."
        )
        return [value]
    elif isinstance(value, Iterable):
        return [str(item) for item in value]
    else:
        raise log_error(
            ValueError, f"Invalid value ({value}). Expected a list of strings."
        )


def _ensure_type_ndarray(value: Any) -> None:
    """Raise ValueError if the value is not a numpy array."""
    if not isinstance(value, np.ndarray):
        raise log_error(
            ValueError, f"Expected a numpy array, but got {type(value)}."
        )


def _set_fps_to_none_if_invalid(fps: Optional[float]) -> Optional[float]:
    """Set fps to None if a non-positive float is passed."""
    if fps is not None and fps <= 0:
        log_warning(
            f"Invalid fps value ({fps}). Expected a positive number. "
            "Setting fps to None."
        )
        return None
    return fps


def _validate_list_length(
    attribute: str, value: Optional[list], expected_length: int
):
    """Raise a ValueError if the list does not have the expected length."""
    if (value is not None) and (len(value) != expected_length):
        raise log_error(
            ValueError,
            f"Expected `{attribute}` to have length {expected_length}, "
            f"but got {len(value)}.",
        )


@define(kw_only=True)
class ValidPosesDataset:
    """Class for validating data intended for a ``movement`` dataset.

    Attributes
    ----------
    position_array : np.ndarray
        Array of shape (n_frames, n_individuals, n_keypoints, n_space)
        containing the poses.
    confidence_array : np.ndarray, optional
        Array of shape (n_frames, n_individuals, n_keypoints) containing
        the point-wise confidence scores.
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
    source_software : str, optional
        Name of the software from which the poses were loaded.
        Defaults to None.

    """

    # Define class attributes
    position_array: np.ndarray = field()
    confidence_array: Optional[np.ndarray] = field(default=None)
    individual_names: Optional[list[str]] = field(
        default=None,
        converter=converters.optional(_list_of_str),
    )
    keypoint_names: Optional[list[str]] = field(
        default=None,
        converter=converters.optional(_list_of_str),
    )
    fps: Optional[float] = field(
        default=None,
        converter=converters.pipe(  # type: ignore
            converters.optional(float), _set_fps_to_none_if_invalid
        ),
    )
    source_software: Optional[str] = field(
        default=None,
        validator=validators.optional(validators.instance_of(str)),
    )

    # Add validators
    @position_array.validator
    def _validate_position_array(self, attribute, value):
        _ensure_type_ndarray(value)
        if value.ndim != 4:
            raise log_error(
                ValueError,
                f"Expected `{attribute}` to have 4 dimensions, "
                f"but got {value.ndim}.",
            )
        if value.shape[-1] not in [2, 3]:
            raise log_error(
                ValueError,
                f"Expected `{attribute}` to have 2 or 3 spatial dimensions, "
                f"but got {value.shape[-1]}.",
            )

    @confidence_array.validator
    def _validate_confidence_array(self, attribute, value):
        if value is not None:
            _ensure_type_ndarray(value)
            expected_shape = self.position_array.shape[:-1]
            if value.shape != expected_shape:
                raise log_error(
                    ValueError,
                    f"Expected `{attribute}` to have shape "
                    f"{expected_shape}, but got {value.shape}.",
                )

    @individual_names.validator
    def _validate_individual_names(self, attribute, value):
        if self.source_software == "LightningPose":
            # LightningPose only supports a single individual
            _validate_list_length(attribute, value, 1)
        else:
            _validate_list_length(
                attribute, value, self.position_array.shape[1]
            )

    @keypoint_names.validator
    def _validate_keypoint_names(self, attribute, value):
        _validate_list_length(attribute, value, self.position_array.shape[2])

    def __attrs_post_init__(self):
        """Assign default values to optional attributes (if None)."""
        if self.confidence_array is None:
            self.confidence_array = np.full(
                (self.position_array.shape[:-1]), np.nan, dtype="float32"
            )
            log_warning(
                "Confidence array was not provided."
                "Setting to an array of NaNs."
            )
        if self.individual_names is None:
            self.individual_names = [
                f"individual_{i}" for i in range(self.position_array.shape[1])
            ]
            log_warning(
                "Individual names were not provided. "
                f"Setting to {self.individual_names}."
            )
        if self.keypoint_names is None:
            self.keypoint_names = [
                f"keypoint_{i}" for i in range(self.position_array.shape[2])
            ]
            log_warning(
                "Keypoint names were not provided. "
                f"Setting to {self.keypoint_names}."
            )
