"""``attrs`` classes for validating data structures."""

from collections.abc import Iterable
from typing import Any

import numpy as np
from attrs import converters, define, field, validators

from movement.utils.logging import log_error, log_warning


def _list_of_str(value: str | Iterable[Any]) -> list[str]:
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
    """Raise ValueError the value is a not numpy array."""
    if not isinstance(value, np.ndarray):
        raise log_error(
            ValueError, f"Expected a numpy array, but got {type(value)}."
        )


def _set_fps_to_none_if_invalid(fps: float | None) -> float | None:
    """Set fps to None if a non-positive float is passed."""
    if fps is not None and fps <= 0:
        log_warning(
            f"Invalid fps value ({fps}). Expected a positive number. "
            "Setting fps to None."
        )
        return None
    return fps


def _validate_list_length(attribute, value: list | None, expected_length: int):
    """Raise a ValueError if the list does not have the expected length."""
    if (value is not None) and (len(value) != expected_length):
        raise log_error(
            ValueError,
            f"Expected '{attribute.name}' to have length {expected_length}, "
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
    confidence_array: np.ndarray | None = field(default=None)
    individual_names: list[str] | None = field(
        default=None,
        converter=converters.optional(_list_of_str),
    )
    keypoint_names: list[str] | None = field(
        default=None,
        converter=converters.optional(_list_of_str),
    )
    fps: float | None = field(
        default=None,
        converter=converters.pipe(  # type: ignore
            converters.optional(float), _set_fps_to_none_if_invalid
        ),
    )
    source_software: str | None = field(
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
                f"Expected '{attribute.name}' to have 4 dimensions, "
                f"but got {value.ndim}.",
            )
        if value.shape[-1] not in [2, 3]:
            raise log_error(
                ValueError,
                f"Expected '{attribute.name}' to have 2 or 3 spatial "
                f"dimensions, but got {value.shape[-1]}.",
            )

    @confidence_array.validator
    def _validate_confidence_array(self, attribute, value):
        if value is not None:
            _ensure_type_ndarray(value)
            expected_shape = self.position_array.shape[:-1]
            if value.shape != expected_shape:
                raise log_error(
                    ValueError,
                    f"Expected '{attribute.name}' to have shape "
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


@define(kw_only=True)
class ValidBboxesDataset:
    """Class for validating bounding boxes' data for a ``movement`` dataset.

    We consider 2D bounding boxes only.

    Attributes
    ----------
    position_array : np.ndarray
        Array of shape (n_frames, n_individual_names, n_space)
        containing the bounding boxes' centroid positions.
    shape_array : np.ndarray
        Array of shape (n_frames, n_individual_names, n_space)
        containing the bounding boxes' width (extent along the
        x-axis) and height (extent along the y-axis).
    confidence_array : np.ndarray, optional
        Array of shape (n_frames, n_individuals, n_keypoints) containing
        the bounding boxes' confidence scores. If None (default), the
        confidence scores will be set to an array of NaNs.
    individual_names : list of str, optional
        List of individual names for the tracked bounding boxes in the video.
        If None (default), bounding boxes are assigned names based on the size
        of the `position_array`. The names will be in the format of `id_<N>`,
        where <N>  is an integer from 1 to `position_array.shape[1]`.
    fps : float, optional
        Frames per second defining the sampling rate of the data.
        Defaults to None.
    source_software : str, optional
        Name of the software from which the bounding boxes were loaded.
        Defaults to None.

    """

    # Required attributes
    position_array: np.ndarray = field()
    shape_array: np.ndarray = field()

    # Optional attributes
    confidence_array: np.ndarray | None = field(default=None)
    individual_names: list[str] | None = field(
        default=None,
        converter=converters.optional(
            _list_of_str
        ),  # force into list of strings if not
    )
    fps: float | None = field(
        default=None,
        converter=converters.pipe(  # type: ignore
            converters.optional(float), _set_fps_to_none_if_invalid
        ),
    )
    source_software: str | None = field(
        default=None,
        validator=validators.optional(validators.instance_of(str)),
    )

    # Validators
    @position_array.validator
    @shape_array.validator
    def _validate_position_and_shape_arrays(self, attribute, value):
        _ensure_type_ndarray(value)

        # check last dimension (spatial) has 2 coordinates
        n_expected_spatial_coordinates = 2
        if value.shape[-1] != n_expected_spatial_coordinates:
            raise log_error(
                ValueError,
                f"Expected '{attribute.name}' to have 2 spatial coordinates, "
                f"but got {value.shape[-1]}.",
            )

    @individual_names.validator
    def _validate_individual_names(self, attribute, value):
        if value is not None:
            _validate_list_length(
                attribute, value, self.position_array.shape[1]
            )

            # check n_individual_names are unique
            # NOTE: combined with the requirement above, we are enforcing
            # unique IDs per frame
            if len(value) != len(set(value)):
                raise log_error(
                    ValueError,
                    "individual_names passed to the dataset are not unique. "
                    f"There are {len(value)} elements in the list, but "
                    f"only {len(set(value))} are unique.",
                )

    @confidence_array.validator
    def _validate_confidence_array(self, attribute, value):
        if value is not None:
            _ensure_type_ndarray(value)

            expected_shape = self.position_array.shape[:-1]
            if value.shape != expected_shape:
                raise log_error(
                    ValueError,
                    f"Expected '{attribute.name}' to have shape "
                    f"{expected_shape}, but got {value.shape}.",
                )

    # Define defaults
    def __attrs_post_init__(self):
        """Assign default values to optional attributes (if None).

        If no confidence_array is provided, set it to an array of NaNs.
        If no individual names are provided, assign them unique IDs per frame,
        starting with 1 ("id_1")
        """
        if self.confidence_array is None:
            self.confidence_array = np.full(
                (self.position_array.shape[:-1]),
                np.nan,
                dtype="float32",
            )
            log_warning(
                "Confidence array was not provided."
                "Setting to an array of NaNs."
            )

        if self.individual_names is None:
            self.individual_names = [
                f"id_{i+1}" for i in range(self.position_array.shape[1])
            ]
            log_warning(
                "Individual names for the bounding boxes "
                "were not provided. "
                "Setting to 1-based IDs that are unique per frame: \n"
                f"{self.individual_names}.\n"
            )
