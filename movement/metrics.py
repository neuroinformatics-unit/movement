import numpy as np

def trajectory_complexity(x, y):
    """
    Compute the trajectory complexity (d = D / L).
    
    Parameters:
        x (array-like): X-coordinates of the trajectory.
        y (array-like): Y-coordinates of the trajectory.
    
    Returns:
        float: Trajectory complexity measure (D / L), where:
               - D is the Euclidean distance between start and end points.
               - L is the total length of the path.
               - Returns 1 for a straight-line path, and values < 1 for more complex paths.
    """
    if len(x) < 2 or len(y) < 2:
        raise ValueError("Trajectory must contain at least two points.")

    # Compute the straight-line distance (D) between the start and end points
    D = np.sqrt((x[-1] - x[0])**2 + (y[-1] - y[0])**2)

    # Compute the total trajectory length (L)
    L = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))

    # Avoid division by zero
    return D / L if L > 0 else 0.0
