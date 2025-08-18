"""``attrs`` classes for validating data structures."""

import warnings
from collections.abc import Iterable
from typing import Any, ClassVar

import attrs
import numpy as np
from attrs import converters, define, field, validators

from movement.utils.logging import logger


def _convert_to_list_of_str(value: str | Iterable[Any]) -> list[str]:
    """Try to coerce the value into a list of strings."""
    if isinstance(value, str):
        warnings.warn(
            f"Invalid value ({value}). Expected a list of strings. "
            "Converting to a list of length 1.",
            UserWarning,
            stacklevel=2,
        )
        return [value]
    elif isinstance(value, Iterable):
        return [str(item) for item in value]
    else:
        raise logger.error(
            ValueError(f"Invalid value ({value}). Expected a list of strings.")
        )


def _convert_fps_to_none_if_invalid(fps: float | None) -> float | None:
    """Set fps to None if a non-positive float is passed."""
    if fps is not None and fps <= 0:
        warnings.warn(
            f"Invalid fps value ({fps}). Expected a positive number. "
            "Setting fps to None.",
            UserWarning,
            stacklevel=2,
        )
        return None
    return fps


def _validate_type_ndarray(value: Any) -> None:
    """Raise ValueError the value is a not numpy array."""
    if not isinstance(value, np.ndarray):
        raise logger.error(
            ValueError(f"Expected a numpy array, but got {type(value)}.")
        )


def _validate_array_shape(
    attribute: attrs.Attribute, value: np.ndarray, expected_shape: tuple
):
    """Raise ValueError if the value does not have the expected shape."""
    if value.shape != expected_shape:
        raise logger.error(
            ValueError(
                f"Expected '{attribute.name}' to have shape {expected_shape}, "
                f"but got {value.shape}."
            )
        )


def _validate_list_length(
    attribute: attrs.Attribute, value: list | None, expected_length: int
):
    """Raise a ValueError if the list does not have the expected length."""
    if (value is not None) and (len(value) != expected_length):
        raise logger.error(
            ValueError(
                f"Expected '{attribute.name}' to have "
                f"length {expected_length}, but got {len(value)}."
            )
        )


@define(kw_only=True)
class ValidPosesDataset:
    """Class for validating poses data intended for a ``movement`` dataset.

    The validator ensures that within the ``movement poses`` dataset:

    - The required ``position_array`` is a numpy array
      with the ``space`` dimension containing 2 or 3 spatial coordinates.
    - The optional ``confidence_array``, if provided, is a numpy array
      with its shape matching that of the ``position_array``,
      excluding the ``space`` dimension;
      otherwise, it defaults to an array of NaNs.
    - The optional ``individual_names`` and ``keypoint_names``,
      if provided, match the number of individuals and keypoints
      in the dataset, respectively; otherwise, default names are assigned.
    - The optional ``fps`` is a positive float; otherwise, it defaults to None.
    - The optional ``source_software`` is a string; otherwise,
      it defaults to None.

    Attributes
    ----------
    position_array : np.ndarray
        Array of shape (n_frames, n_space, n_keypoints, n_individuals)
        containing the poses.
    confidence_array : np.ndarray, optional
        Array of shape (n_frames, n_keypoints, n_individuals) containing
        the point-wise confidence scores.
        If None (default), the scores will be set to an array of NaNs.
    individual_names : list of str, optional
        List of unique names for the individuals in the video. If None
        (default), the individuals will be named "id_0", "id_1", etc.
    keypoint_names : list of str, optional
        List of unique names for the keypoints in the skeleton. If None
        (default), the keypoints will be named "keypoint_0", "keypoint_1",
        etc.
    fps : float, optional
        Frames per second of the video. Defaults to None.
    source_software : str, optional
        Name of the software from which the poses were loaded.
        Defaults to None.

    Raises
    ------
    ValueError
        If the dataset does not meet the ``movement poses``
        dataset requirements.

    """

    # Required attributes
    position_array: np.ndarray = field()

    # Optional attributes
    confidence_array: np.ndarray | None = field(default=None)
    individual_names: list[str] | None = field(
        default=None,
        converter=converters.optional(_convert_to_list_of_str),
    )
    keypoint_names: list[str] | None = field(
        default=None,
        converter=converters.optional(_convert_to_list_of_str),
    )
    fps: float | None = field(
        default=None,
        converter=converters.pipe(  # type: ignore
            converters.optional(float), _convert_fps_to_none_if_invalid
        ),
    )
    source_software: str | None = field(
        default=None,
        validator=validators.optional(validators.instance_of(str)),
    )

    # Class variables
    DIM_NAMES: ClassVar[tuple] = ("time", "space", "keypoints", "individuals")
    VAR_NAMES: ClassVar[tuple] = ("position", "confidence")

    # Add validators
    @position_array.validator
    def _validate_position_array(self, attribute, value):
        _validate_type_ndarray(value)
        n_dims = value.ndim
        if n_dims != 4:
            raise logger.error(
                ValueError(
                    f"Expected '{attribute.name}' to have 4 dimensions, "
                    f"but got {n_dims}."
                )
            )
        space_dim_shape = value.shape[1]
        if space_dim_shape not in [2, 3]:
            raise logger.error(
                ValueError(
                    f"Expected '{attribute.name}' to have 2 or 3 spatial "
                    f"dimensions, but got {space_dim_shape}."
                )
            )

    @confidence_array.validator
    def _validate_confidence_array(self, attribute, value):
        if value is not None:
            _validate_type_ndarray(value)
            # Expected shape is the same as position_array,
            # but without the `space` dim
            expected_shape = (
                self.position_array.shape[:1] + self.position_array.shape[2:]
            )
            _validate_array_shape(
                attribute, value, expected_shape=expected_shape
            )

    @individual_names.validator
    def _validate_individual_names(self, attribute, value):
        if self.source_software == "LightningPose":
            # LightningPose only supports a single individual
            _validate_list_length(attribute, value, 1)
        else:
            _validate_list_length(
                attribute, value, self.position_array.shape[-1]
            )

    @keypoint_names.validator
    def _validate_keypoint_names(self, attribute, value):
        _validate_list_length(attribute, value, self.position_array.shape[2])

    def __attrs_post_init__(self):
        """Assign default values to optional attributes (if None)."""
        position_array_shape = self.position_array.shape
        if self.confidence_array is None:
            self.confidence_array = np.full(
                position_array_shape[:1] + position_array_shape[2:],
                np.nan,
                dtype="float32",
            )
            logger.warning(
                "Confidence array was not provided."
                "Setting to an array of NaNs."
            )
        if self.individual_names is None:
            self.individual_names = [
                f"id_{i}" for i in range(position_array_shape[-1])
            ]
            logger.warning(
                "Individual names were not provided. "
                f"Setting to {self.individual_names}."
            )
        if self.keypoint_names is None:
            self.keypoint_names = [
                f"keypoint_{i}" for i in range(position_array_shape[2])
            ]
            logger.warning(
                "Keypoint names were not provided. "
                f"Setting to {self.keypoint_names}."
            )


@define(kw_only=True)
class ValidBboxesDataset:
    """Class for validating bounding boxes data for a ``movement`` dataset.

    The validator considers 2D bounding boxes only. It ensures that
    within the ``movement bboxes`` dataset:

    - The required ``position_array`` and ``shape_array`` are numpy arrays,
      with the ``space`` dimension containing 2 spatial coordinates.
    - The optional ``confidence_array``, if provided, is a numpy array
      with its shape matching that of the ``position_array``,
      excluding the ``space`` dimension;
      otherwise, it defaults to an array of NaNs.
    - The optional ``individual_names``, if provided, match the number of
      individuals in the dataset; otherwise, default names are assigned.
    - The optional ``frame_array``, if provided, is a column vector
      with the frame numbers; otherwise, it defaults to an array of
      0-based integers.
    - The optional ``fps`` is a positive float; otherwise, it defaults to None.
    - The optional ``source_software`` is a string; otherwise, it defaults to
      None.

    Attributes
    ----------
    position_array : np.ndarray
        Array of shape (n_frames, n_space, n_individuals)
        containing the tracks of the bounding box centroids.
    shape_array : np.ndarray
        Array of shape (n_frames, n_space, n_individuals)
        containing the shape of the bounding boxes. The shape of a bounding
        box is its width (extent along the x-axis of the image) and height
        (extent along the y-axis of the image).
    confidence_array : np.ndarray, optional
        Array of shape (n_frames, n_individuals) containing
        the confidence scores of the bounding boxes. If None (default), the
        confidence scores are set to an array of NaNs.
    individual_names : list of str, optional
        List of individual names for the tracked bounding boxes in the video.
        If None (default), bounding boxes are assigned names based on the size
        of the ``position_array``. The names will be in the format of
        ``id_<N>``, where <N>  is an integer from 0 to
        ``position_array.shape[1]-1``.
    frame_array : np.ndarray, optional
        Array of shape (n_frames, 1) containing the frame numbers for which
        bounding boxes are defined. If None (default), frame numbers will
        be assigned based on the first dimension of the ``position_array``,
        starting from 0.
    fps : float, optional
        Frames per second defining the sampling rate of the data.
        Defaults to None.
    source_software : str, optional
        Name of the software that generated the data. Defaults to None.

    Raises
    ------
    ValueError
        If the dataset does not meet the ``movement bboxes`` dataset
        requirements.

    """

    # Required attributes
    position_array: np.ndarray = field()
    shape_array: np.ndarray = field()

    # Optional attributes
    confidence_array: np.ndarray | None = field(default=None)
    individual_names: list[str] | None = field(
        default=None,
        converter=converters.optional(
            _convert_to_list_of_str
        ),  # force into list of strings if not
    )
    frame_array: np.ndarray | None = field(default=None)
    fps: float | None = field(
        default=None,
        converter=converters.pipe(  # type: ignore
            converters.optional(float), _convert_fps_to_none_if_invalid
        ),
    )
    source_software: str | None = field(
        default=None,
        validator=validators.optional(validators.instance_of(str)),
    )

    DIM_NAMES: ClassVar[tuple] = ("time", "space", "individuals")
    VAR_NAMES: ClassVar[tuple] = ("position", "shape", "confidence")

    # Validators
    @position_array.validator
    @shape_array.validator
    def _validate_position_and_shape_arrays(self, attribute, value):
        _validate_type_ndarray(value)
        # check `space` dim (at idx 1) has 2 coordinates
        n_expected_spatial_coordinates = 2
        n_spatial_coordinates = value.shape[1]
        if n_spatial_coordinates != n_expected_spatial_coordinates:
            raise logger.error(
                ValueError(
                    f"Expected '{attribute.name}' to have "
                    f"{n_expected_spatial_coordinates} spatial coordinates, "
                    f"but got {n_spatial_coordinates}."
                )
            )

    @individual_names.validator
    def _validate_individual_names(self, attribute, value):
        if value is not None:
            _validate_list_length(
                attribute, value, self.position_array.shape[-1]
            )
            # check n_individual_names are unique
            # NOTE: combined with the requirement above, we are enforcing
            # unique IDs per frame
            if len(value) != len(set(value)):
                raise logger.error(
                    ValueError(
                        "individual_names are not unique. "
                        f"There are {len(value)} elements in the list, but "
                        f"only {len(set(value))} are unique."
                    )
                )

    @confidence_array.validator
    def _validate_confidence_array(self, attribute, value):
        if value is not None:
            _validate_type_ndarray(value)
            # Expected shape is the same as position_array,
            # but without the `space` dim
            expected_shape = (
                self.position_array.shape[:1] + self.position_array.shape[2:]
            )
            _validate_array_shape(
                attribute, value, expected_shape=expected_shape
            )

    @frame_array.validator
    def _validate_frame_array(self, attribute, value):
        if value is not None:
            _validate_type_ndarray(value)
            # should be a column vector (n_frames, 1)
            _validate_array_shape(
                attribute,
                value,
                expected_shape=(self.position_array.shape[0], 1),
            )
            # check frames are monotonically increasing
            if not np.all(np.diff(value, axis=0) >= 1):
                raise logger.error(
                    ValueError(
                        f"Frame numbers in {attribute.name} are "
                        "not monotonically increasing."
                    )
                )

    # Define defaults
    def __attrs_post_init__(self):
        """Assign default values to optional attributes (if None).

        If no confidence_array is provided, set it to an array of NaNs.
        If no individual names are provided, assign them unique IDs per frame,
        starting with 0 ("id_0").
        """
        position_array_shape = self.position_array.shape
        # assign default confidence_array
        if self.confidence_array is None:
            self.confidence_array = np.full(
                position_array_shape[:1] + position_array_shape[2:],
                np.nan,
                dtype="float32",
            )
            logger.warning(
                "Confidence array was not provided. "
                "Setting to an array of NaNs."
            )
        # assign default individual_names
        if self.individual_names is None:
            self.individual_names = [
                f"id_{i}" for i in range(position_array_shape[-1])
            ]
            logger.warning(
                "Individual names for the bounding boxes "
                "were not provided. "
                "Setting to 0-based IDs that are unique per frame: \n"
                f"{self.individual_names}.\n"
            )
        # assign default frame_array
        if self.frame_array is None:
            n_frames = position_array_shape[0]
            self.frame_array = np.arange(n_frames).reshape(-1, 1)
            logger.warning(
                "Frame numbers were not provided. "
                "Setting to an array of 0-based integers."
            )
