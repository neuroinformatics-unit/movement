import numpy as np
from collections import defaultdict

class Individual:
    def __init__(self, name, keypoints, keypoint_names):
        """
        keypoints: np.ndarray of shape (n_frames, n_keypoints, 2)
        keypoint_names: list of keypoint names specific to this individual
        """
        self.name = name
        self.keypoints = keypoints
        self.keypoint_names = keypoint_names

    def get_keypoint(self, name):
        """Returns the keypoint track for a given name."""
        if name not in self.keypoint_names:
            raise ValueError(f"Keypoint {name} not found for individual {self.name}")
        idx = self.keypoint_names.index(name)
        return self.keypoints[:, idx, :]


class Scene:
    def __init__(self):
        self.individuals = {}

    def add_individual(self, individual: Individual):
        self.individuals[individual.name] = individual

    def get_common_keypoints(self):
        """Returns a list of keypoints that are shared across all individuals."""
        if not self.individuals:
            return []

        keypoint_sets = [set(ind.keypoint_names) for ind in self.individuals.values()]
        return list(set.intersection(*keypoint_sets))

    def get_keypoint_across_individuals(self, keypoint_name):
        """Return the specified keypoint for all individuals who have it."""
        result = {}
        for name, ind in self.individuals.items():
            if keypoint_name in ind.keypoint_names:
                result[name] = ind.get_keypoint(keypoint_name)
        return result
