"""Define the canonical structure of Movement Datasets."""


class MovementDataset:
    """Base class to define the canonical structure of a Movement Dataset."""

    # Base dimensions and variables common to all datasets
    DIM_NAMES: tuple[str, ...] = ("time", "space")
    VAR_NAMES: tuple[str, ...] = ("position", "confidence")

    @classmethod
    def get_dim_names(cls):
        """Get dimension names for the dataset."""
        return cls.DIM_NAMES

    @classmethod
    def get_var_names(cls):
        """Get variable names for the dataset."""
        return cls.VAR_NAMES


class PosesDataset(MovementDataset):
    """Dataset class for pose data, extending MovementDataset."""

    # Additional dimensions and variables specific to poses
    DIM_NAMES: tuple[str, ...] = ("time", "individuals", "keypoints", "space")
    VAR_NAMES: tuple[str, ...] = MovementDataset.VAR_NAMES


class BboxesDataset(MovementDataset):
    """Dataset class for bounding boxes' data, extending MovementDataset."""

    # Additional dimensions and variables specific to bounding boxes
    DIM_NAMES: tuple[str, ...] = ("time", "individuals", "space")
    VAR_NAMES: tuple[str, ...] = ("position", "shape", "confidence")
