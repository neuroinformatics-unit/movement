import numpy as np

def trajectory_length(traj):
    return np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))

def trajectory_variance(traj):
    return np.var(traj, axis=0).sum()