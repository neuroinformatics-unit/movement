import numpy as np

from movement.kinematics.trajectory_metrics import trajectory_length


def test_trajectory_length():
    traj = np.array([[0, 0], [3, 4]])
    assert trajectory_length(traj) == 5.0
