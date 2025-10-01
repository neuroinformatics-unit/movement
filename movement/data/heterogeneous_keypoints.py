"""Module for handling individuals with heterogeneous keypoints and scenes."""


class Individual:
    """Represents an individual with a specific set of keypoints."""

    def __init__(self, name, keypoints, keypoint_names):
        """Initialize an Individual.

        Parameters
        ----------
        name : str
            Name of the individual.
        keypoints : np.ndarray
            Array of shape (n_frames, n_keypoints, 2)
            representing keypoint positions.
        keypoint_names : list of str
            Names of the keypoints specific to this individual.

        """
        if keypoints.ndim != 3 or keypoints.shape[2] != 2:
            raise ValueError(
                "Keypoints must be of shape (frames, keypoints, 2). "
                f"Got {keypoints.shape}"
            )
        if keypoints.shape[1] != len(keypoint_names):
            raise ValueError(
                f"Number of keypoint names ({len(keypoint_names)}) "
                f"does not match keypoints array ({keypoints.shape[1]})"
            )

        self.name = name
        self.keypoints = keypoints
        self.keypoint_names = keypoint_names

    def get_keypoint(self, name):
        """Return the keypoint track for a given name.

        Parameters
        ----------
        name : str
            Name of the keypoint.

        Returns
        -------
        np.ndarray
            Keypoint positions across frames.

        Raises
        ------
        ValueError
            If the keypoint name is not found.

        """
        if name not in self.keypoint_names:
            raise ValueError(
                f"Keypoint {name} not found for individual {self.name}"
            )
        idx = self.keypoint_names.index(name)
        return self.keypoints[:, idx, :]


class Scene:
    """Manages multiple individuals and supports querying shared keypoints."""

    def __init__(self):
        """Initialize an empty Scene."""
        self.individuals = {}

    def add_individual(self, individual: Individual):
        """Add an individual to the scene.

        Parameters
        ----------
        individual : Individual
            The individual to be added.

        """
        self.individuals[individual.name] = individual

    def get_common_keypoints(self):
        """Return a list of keypoints shared across all individuals.

        Returns
        -------
        list of str
            Keypoint names that are common to all individuals.

        """
        if not self.individuals:
            return []
        keypoint_sets = [
            set(ind.keypoint_names) for ind in self.individuals.values()
        ]
        return list(set.intersection(*keypoint_sets))

    def get_keypoint_across_individuals(self, keypoint_name):
        """Return the specified keypoint for all individuals who have it.

        Parameters
        ----------
        keypoint_name : str
            Name of the keypoint to retrieve.

        Returns
        -------
        dict
            Dictionary mapping individual names to keypoint tracks
            (np.ndarray).

        """
        result = {}
        for name, ind in self.individuals.items():
            if keypoint_name in ind.keypoint_names:
                result[name] = ind.get_keypoint(keypoint_name)
        return result
