"""``attrs`` classes for validating data structures."""

import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

import attrs
import numpy as np
import xarray as xr
from attrs import converters, define, field, validators
from numpy.typing import NDArray

from movement.utils.logging import logger

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _convert_to_list_of_str(value: str | Iterable[Any]) -> list[str]:
    """Try to coerce the value into a list of strings."""
    if isinstance(value, str):
        warnings.warn(
            f"Expected a list of strings, but got a string ({value}). "
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
class _BaseDatasetInputs(ABC):
    """Abstract base class for validating ``movement`` dataset inputs.

    This base class centralises shared fields, validators, and default
    assignment logic for creating ``movement`` datasets
    (e.g. poses, bounding boxes).
    It registers the attrs validators for required fields like
    ``position_array`` and optional fields like ``confidence_array`` and
    ``individual_names``.
    Subclasses must implement ``to_dataset()`` and define class variables
    ``DIM_NAMES``, ``VAR_NAMES``, and ``_ALLOWED_SPACE_DIM_SIZE``.
    """

    # --- Required fields ---
    position_array: np.ndarray = field(
        validator=validators.instance_of(np.ndarray)
    )
    # --- Optional fields ---
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
    # --- Required class variables (to be defined by subclasses) ---
    DIM_NAMES: ClassVar[tuple[str, ...]]
    VAR_NAMES: ClassVar[tuple[str, ...]]
    _ALLOWED_SPACE_DIM_SIZE: ClassVar[int | Iterable[int]]

    # --- Lifecycle hooks ---
    def __attrs_post_init__(self):
        """Assign default values to optional attributes (if None)."""
        # confidence_array default: array of NaNs with appropriate shape
        if self.confidence_array is None:
            self.confidence_array = np.full(
                self._confidence_expected_shape, np.nan, dtype="float32"
            )
            logger.info(
                "Confidence array was not provided."
                "Setting to an array of NaNs."
            )
        # individual_names default: id_0, id_1, ...
        if self.individual_names is None and "individual" in self.DIM_NAMES:
            n_inds = self.position_array.shape[
                self.DIM_NAMES.index("individual")
            ]
            self.individual_names = [f"id_{i}" for i in range(n_inds)]
            logger.info(
                "Individual names were not provided. "
                f"Setting to {self.individual_names}."
            )

    # --- Properties (derived attributes) ---
    @property
    def _confidence_expected_shape(self):
        """Return expected shape for confidence_array."""
        # confidence shape == position_array shape without the space dim
        return tuple(
            dim
            for i, dim in enumerate(self.position_array.shape)
            if i != self.DIM_NAMES.index("space")
        )

    # --- Validators ---
    @position_array.validator
    def _validate_position_array(self, attribute, value):
        """Raise ValueError if array dimensions are unexpected."""
        # Check array dimensions match the number of DIM_NAMES
        expected_ndim = len(self.DIM_NAMES)
        if value.ndim != expected_ndim:
            raise logger.error(
                ValueError(
                    f"Expected '{attribute.name}' to have "
                    f"{expected_ndim} dimensions, but got {value.ndim}."
                )
            )
        # Check size of 'space' dimension
        allowed_axis_size = self._ALLOWED_SPACE_DIM_SIZE
        space_dim_size = value.shape[self.DIM_NAMES.index("space")]
        if not isinstance(allowed_axis_size, Iterable):
            allowed_axis_size = (allowed_axis_size,)
        if space_dim_size not in allowed_axis_size:
            allowed_dims_str = " or ".join(
                str(dim) for dim in allowed_axis_size
            )
            raise logger.error(
                ValueError(
                    f"Expected '{attribute.name}' to have {allowed_dims_str} "
                    f"spatial dimensions, but got {space_dim_size}."
                )
            )

    @confidence_array.validator
    def _validate_confidence_array(self, attribute, value):
        """Check confidence_array type and shape."""
        if value is not None:
            expected_shape = self._confidence_expected_shape
            self._validate_array_shape(
                attribute, value, expected_shape=expected_shape
            )

    @individual_names.validator
    def _validate_individual_names(self, attribute, value):
        """Validate individual_names length and uniqueness."""
        if value is not None:
            individuals_dim_index = self.DIM_NAMES.index("individual")
            self._validate_list_length(
                attribute,
                value,
                self.position_array.shape[individuals_dim_index],
            )
            self._validate_list_uniqueness(attribute, value)

    # --- Utility methods ---
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
    def _validate_list_length(
        attribute: attrs.Attribute, value: list | None, expected_length: int
    ):
        """Raise a ValueError if the list does not have the expected length."""
        if value is not None and len(value) != expected_length:
            raise logger.error(
                ValueError(
                    f"Expected '{attribute.name}' to have "
                    f"length {expected_length}, but got {len(value)}."
                )
            )

    @staticmethod
    def _validate_list_uniqueness(
        attribute: attrs.Attribute, value: list | None
    ):
        """Raise a ValueError if the list does not have unique elements."""
        if value is not None and len(value) != len(set(value)):
            raise logger.error(
                ValueError(
                    f"Elements in '{attribute.name}' are not unique. "
                    f"There are {len(value)} elements in the list, but "
                    f"only {len(set(value))} are unique."
                )
            )

    @abstractmethod
    def to_dataset(self) -> xr.Dataset:
        """Convert validated inputs to a ``movement`` xarray.Dataset.

        Returns
        -------
        xarray.Dataset
            ``movement`` dataset containing the validated data and metadata.

        """
        ...

    @classmethod
    def validate(cls, ds: xr.Dataset) -> None:
        """Validate that the dataset has the required variables and dimensions.

        Parameters
        ----------
        ds : xarray.Dataset
            Dataset to validate.

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
        missing_vars = set(cls.VAR_NAMES) - set(
            cast("Iterable[str]", ds.data_vars.keys())
        )
        if missing_vars:
            raise logger.error(
                ValueError(
                    f"Missing required data variables: {sorted(missing_vars)}"
                )
            )  # sort for a reproducible error message
        # Ignore type error - ds.dims will soon return a set of dim names
        missing_dims = set(cls.DIM_NAMES) - set(ds.dims)  # type: ignore[arg-type]
        if missing_dims:
            raise logger.error(
                ValueError(
                    f"Missing required dimensions: {sorted(missing_dims)}"
                )
            )  # sort for a reproducible error message


@define(kw_only=True)
class ValidPosesInputs(_BaseDatasetInputs):
    """Class for validating input data for a ``movement poses`` dataset.

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
        "keypoint",
        "individual",
    )
    VAR_NAMES: ClassVar[tuple[str, ...]] = ("position", "confidence")
    _ALLOWED_SPACE_DIM_SIZE: ClassVar[Iterable[int]] = (2, 3)

    @keypoint_names.validator
    def _validate_keypoint_names(self, attribute, value):
        """Validate keypoint_names length and uniqueness."""
        keypoints_dim_index = self.DIM_NAMES.index("keypoint")
        self._validate_list_length(
            attribute, value, self.position_array.shape[keypoints_dim_index]
        )
        self._validate_list_uniqueness(attribute, value)

    def __attrs_post_init__(self):
        """Assign default values to optional attributes (if None)."""
        super().__attrs_post_init__()
        position_array_shape = self.position_array.shape
        keypoints_dim_index = self.DIM_NAMES.index("keypoint")
        if self.keypoint_names is None:
            self.keypoint_names = [
                f"keypoint_{i}"
                for i in range(position_array_shape[keypoints_dim_index])
            ]
            logger.info(
                "Keypoint names were not provided. "
                f"Setting to {self.keypoint_names}."
            )

    def to_dataset(self) -> xr.Dataset:
        """Convert validated poses inputs to a ``movement poses`` dataset.

        Returns
        -------
        xarray.Dataset
            ``movement`` dataset containing the pose tracks, confidence scores,
            and associated metadata.

        """
        n_frames = self.position_array.shape[0]
        n_space = self.position_array.shape[1]
        dataset_attrs: dict[str, str | float | None] = {
            "source_software": self.source_software,
            "ds_type": "poses",
        }
        # Create the time coordinate, depending on the value of fps
        time_coords: NDArray[np.floating] | NDArray[np.integer]
        time_unit: Literal["seconds", "frames"]
        if self.fps is not None:
            time_coords = np.arange(n_frames, dtype=np.float64) / self.fps
            time_unit = "seconds"
            dataset_attrs["fps"] = self.fps
        else:
            time_coords = np.arange(n_frames, dtype=np.int64)
            time_unit = "frames"
        dataset_attrs["time_unit"] = time_unit
        DIM_NAMES = self.DIM_NAMES
        # Convert data to an xarray.Dataset
        return xr.Dataset(
            data_vars={
                "position": xr.DataArray(self.position_array, dims=DIM_NAMES),
                "confidence": xr.DataArray(
                    self.confidence_array, dims=DIM_NAMES[:1] + DIM_NAMES[2:]
                ),
            },
            coords={
                DIM_NAMES[0]: time_coords,
                DIM_NAMES[1]: ["x", "y", "z"][:n_space],
                DIM_NAMES[2]: self.keypoint_names,
                DIM_NAMES[3]: self.individual_names,
            },
            attrs=dataset_attrs,
        )


@define(kw_only=True)
class ValidBboxesInputs(_BaseDatasetInputs):
    """Class for validating input data for a ``movement bboxes`` dataset.

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

    DIM_NAMES: ClassVar[tuple[str, ...]] = ("time", "space", "individual")
    VAR_NAMES: ClassVar[tuple[str, ...]] = ("position", "shape", "confidence")
    _ALLOWED_SPACE_DIM_SIZE: ClassVar[int] = 2

    @shape_array.validator
    def _validate_shape_array(self, attribute, value):
        """Validate shape_array dimensions and shape."""
        super()._validate_position_array(attribute, value)
        # Shape must match that of position_array
        self._validate_array_shape(
            attribute, value, expected_shape=self.position_array.shape
        )

    @frame_array.validator
    def _validate_frame_array(self, attribute, value):
        """Validate frame_array type, shape, and monotonicity."""
        if value is not None:
            # should be a column vector (n_frames, 1)
            time_dim_index = self.DIM_NAMES.index("time")
            self._validate_array_shape(
                attribute,
                value,
                expected_shape=(self.position_array.shape[time_dim_index], 1),
            )
            # check frames are monotonically increasing
            if not np.all(np.diff(value, axis=0) >= 1):
                raise logger.error(
                    ValueError(
                        f"Frame numbers in {attribute.name} are "
                        "not monotonically increasing."
                    )
                )

    def __attrs_post_init__(self):
        """Assign default values to optional attributes (if None)."""
        super().__attrs_post_init__()
        # assign default frame_array
        if self.frame_array is None:
            time_dim_index = self.DIM_NAMES.index("time")
            n_frames = self.position_array.shape[time_dim_index]
            self.frame_array = np.arange(n_frames).reshape(-1, 1)
            logger.info(
                "Frame numbers were not provided. "
                "Setting to an array of 0-based integers."
            )

    def to_dataset(self) -> xr.Dataset:
        """Convert validated bboxes inputs to a ``movement bboxes`` dataset.

        Returns
        -------
        xarray.Dataset
            ``movement`` dataset containing the bounding boxes tracks,
            shapes, confidence scores and associated metadata.

        """
        dataset_attrs: dict[str, str | float | None] = {
            "source_software": self.source_software,
            "ds_type": "bboxes",
        }
        # Ignore type error as ValidBboxesInputs ensures
        # `frame_array` is not None
        time_coords: NDArray[np.floating] | NDArray[np.integer] = (
            self.frame_array.squeeze()  # type: ignore[union-attr]
        )
        time_unit: Literal["seconds", "frames"] = "frames"
        # if fps is provided:
        # time_coords is expressed in seconds, with the time origin
        # set as frame 0 == time 0 seconds
        # Store fps as a dataset attribute
        if self.fps:
            # Compute elapsed time from frame 0.
            time_coords = time_coords / self.fps
            time_unit = "seconds"
            dataset_attrs["fps"] = self.fps
        dataset_attrs["time_unit"] = time_unit
        # Convert data to an xarray.Dataset
        # with dimensions ('time', 'space', 'individuals')
        DIM_NAMES = self.DIM_NAMES
        n_space = self.position_array.shape[1]
        return xr.Dataset(
            data_vars={
                "position": xr.DataArray(self.position_array, dims=DIM_NAMES),
                "shape": xr.DataArray(self.shape_array, dims=DIM_NAMES),
                "confidence": xr.DataArray(
                    self.confidence_array, dims=DIM_NAMES[:1] + DIM_NAMES[2:]
                ),
            },
            coords={
                DIM_NAMES[0]: time_coords,
                DIM_NAMES[1]: ["x", "y", "z"][:n_space],
                DIM_NAMES[2]: self.individual_names,
            },
            attrs=dataset_attrs,
        )
