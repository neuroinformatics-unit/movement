"""``attrs`` classes for validating data structures."""

import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any, ClassVar

import attrs
import numpy as np
import xarray as xr
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


@define(kw_only=True)
class _BaseValidDataset(ABC):
    """Base class for movement dataset validators.

    This abstract class centralises shared fields, validators, and default
    assignment logic for movement datasets (e.g. poses, bounding boxes).
    It registers the attrs validators for required fields like `position_array`
    and optional fields like `confidence_array` and `individual_names`.
    Dataset-specific checks are delegated to subclasses via abstract hooks
    (with suffix `_impl`).
    """

    position_array: np.ndarray = field(
        validator=validators.instance_of(np.ndarray)
    )
    confidence_array: np.ndarray | None = field(
        default=None,
        validator=validators.optional(validators.instance_of(np.ndarray)),
    )
    individual_names: list[str] | None = field(
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

    # Subclasses must define these class variables
    DIM_NAMES: ClassVar[tuple[str, ...]]
    VAR_NAMES: ClassVar[tuple[str, ...]]
    _POSITION_EXPECTED_SPACE_DIM_SIZE: ClassVar[int | Iterable[int]]
    _POSITION_SPACE_AXIS: ClassVar[int] = 1  # Default axis for 'space' dim

    @property
    def _POSITION_EXPECTED_NDIM(self):
        """Return expected number of dimensions for position_array."""
        return len(self.DIM_NAMES)

    @position_array.validator
    def _validate_position_array(self, attribute, value):
        """Check position_array type and dimensions."""
        self._validate_array_dims(
            attribute,
            value,
            self._POSITION_EXPECTED_NDIM,
            self._POSITION_SPACE_AXIS,
            self._POSITION_EXPECTED_SPACE_DIM_SIZE,
        )

    @confidence_array.validator
    def _validate_confidence_array(self, attribute, value):
        """Check confidence_array type and shape."""
        if value is not None:
            expected_shape = self._expected_confidence_shape()
            self._validate_array_shape(
                attribute, value, expected_shape=expected_shape
            )

    @individual_names.validator
    def _validate_individual_names(self, attribute, value):
        """Delegate individual_names validation to subclass implementation."""
        if value is not None:
            self._validate_individual_names_impl(attribute, value)

    def _expected_confidence_shape(self) -> tuple:
        """Return expected shape for confidence_array."""
        # confidence shape == position_array shape without the space dim
        return tuple(
            dim
            for i, dim in enumerate(self.position_array.shape)
            if i != self._POSITION_SPACE_AXIS
        )

    def __attrs_post_init__(self):
        """Assign default values to optional attributes (if None)."""
        # confidence_array default: array of NaNs with appropriate shape
        if self.confidence_array is None:
            self.confidence_array = np.full(
                self._expected_confidence_shape(), np.nan, dtype="float32"
            )
            logger.warning(
                "Confidence array was not provided."
                "Setting to an array of NaNs."
            )
        # individual_names default: id_0, id_1, ...
        if self.individual_names is None:
            n_inds = self.position_array.shape[-1]
            self.individual_names = [f"id_{i}" for i in range(n_inds)]
            logger.warning(
                "Individual names were not provided. "
                f"Setting to {self.individual_names}."
            )

    @abstractmethod
    def _validate_individual_names_impl(self, attribute, value):
        """Perform dataset-specific validation of individual_names."""

    @staticmethod
    def _validate_array_shape(
        attribute: attrs.Attribute, value: np.ndarray, expected_shape: tuple
    ):
        """Raise ValueError if the value does not have the expected shape."""
        if value.shape != expected_shape:
            raise logger.error(
                ValueError(
                    f"Expected '{attribute.name}' to have shape "
                    f"{expected_shape}, but got {value.shape}."
                )
            )

    @staticmethod
    def _validate_array_dims(
        attribute: attrs.Attribute,
        value: np.ndarray,
        expected_ndim: int,
        axis: int,
        expected_axis_size: int | Iterable[int],
    ):
        """Raise ValueError if ndim and/or axis size are unexpected."""
        if value.ndim != expected_ndim:
            raise logger.error(
                ValueError(
                    f"Expected '{attribute.name}' to have "
                    f"{expected_ndim} dimensions, but got {value.ndim}."
                )
            )
        if not isinstance(expected_axis_size, Iterable):
            expected_axis_size = (expected_axis_size,)
        space_dim_size = value.shape[axis]
        if space_dim_size not in expected_axis_size:
            allowed_dims_str = " or ".join(
                str(dim) for dim in expected_axis_size
            )
            raise logger.error(
                ValueError(
                    f"Expected '{attribute.name}' to have {allowed_dims_str} "
                    f"spatial dimensions, but got {space_dim_size}."
                )
            )

    @staticmethod
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
class ValidPosesDataset(_BaseValidDataset):
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

    keypoint_names: list[str] | None = field(
        default=None,
        converter=converters.optional(_convert_to_list_of_str),
    )

    DIM_NAMES: ClassVar[tuple[str, ...]] = (
        "time",
        "space",
        "keypoints",
        "individuals",
    )
    VAR_NAMES: ClassVar[tuple[str, ...]] = ("position", "confidence")
    _POSITION_EXPECTED_SPACE_DIM_SIZE: ClassVar[Iterable[int]] = (2, 3)

    @keypoint_names.validator
    def _validate_keypoint_names(self, attribute, value):
        """Validate keypoint_names length."""
        self._validate_list_length(
            attribute, value, self.position_array.shape[2]
        )

    def _validate_individual_names_impl(self, attribute, value):
        """Validate individual_names length based on source_software."""
        expected_length = (
            1
            if self.source_software == "LightningPose"
            else self.position_array.shape[-1]
        )
        self._validate_list_length(attribute, value, expected_length)

    def __attrs_post_init__(self):
        """Assign default values to optional attributes (if None)."""
        super().__attrs_post_init__()
        position_array_shape = self.position_array.shape
        if self.keypoint_names is None:
            self.keypoint_names = [
                f"keypoint_{i}" for i in range(position_array_shape[2])
            ]
            logger.warning(
                "Keypoint names were not provided. "
                f"Setting to {self.keypoint_names}."
            )


@define(kw_only=True)
class ValidBboxesDataset(_BaseValidDataset):
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

    shape_array: np.ndarray = field(
        validator=validators.instance_of(np.ndarray)
    )
    frame_array: np.ndarray | None = field(
        default=None,
        validator=validators.optional(validators.instance_of(np.ndarray)),
    )

    DIM_NAMES: ClassVar[tuple[str, ...]] = ("time", "space", "individuals")
    VAR_NAMES: ClassVar[tuple[str, ...]] = ("position", "shape", "confidence")
    _POSITION_EXPECTED_SPACE_DIM_SIZE: ClassVar[int] = 2

    @shape_array.validator
    def _validate_shape_array(self, attribute, value):
        """Apply the same validation as position_array."""
        super()._validate_position_array(attribute, value)

    @frame_array.validator
    def _validate_frame_array(self, attribute, value):
        """Validate frame_array type, shape, and monotonicity."""
        if value is not None:
            # should be a column vector (n_frames, 1)
            self._validate_array_shape(
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

    def _validate_individual_names_impl(self, attribute, value):
        """Validate individual_names length and uniqueness."""
        self._validate_list_length(
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

    def __attrs_post_init__(self):
        """Assign default values to optional attributes (if None)."""
        super().__attrs_post_init__()
        # assign default frame_array
        if self.frame_array is None:
            n_frames = self.position_array.shape[0]
            self.frame_array = np.arange(n_frames).reshape(-1, 1)
            logger.warning(
                "Frame numbers were not provided. "
                "Setting to an array of 0-based integers."
            )


def _validate_dataset(
    ds: xr.Dataset,
    dataset_validator: type[ValidBboxesDataset] | type[ValidPosesDataset],
) -> None:
    """Validate the input as a proper ``movement`` dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to validate.
    dataset_validator : type[ValidBboxesDataset] | type[ValidPosesDataset]
        Validator for the dataset.

    Raises
    ------
    TypeError
        If the input is not an xarray Dataset.
    ValueError
        If the dataset is missing required data variables or dimensions
        for a valid ``movement`` dataset.

    """
    if not isinstance(ds, xr.Dataset):
        raise logger.error(
            TypeError(f"Expected an xarray Dataset, but got {type(ds)}.")
        )

    missing_vars = set(dataset_validator.VAR_NAMES) - set(ds.data_vars)
    if missing_vars:
        raise logger.error(
            ValueError(
                f"Missing required data variables: {sorted(missing_vars)}"
            )
        )  # sort for a reproducible error message

    missing_dims = set(dataset_validator.DIM_NAMES) - set(ds.dims)
    if missing_dims:
        raise logger.error(
            ValueError(f"Missing required dimensions: {sorted(missing_dims)}")
        )  # sort for a reproducible error message
