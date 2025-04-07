import matplotlib.pyplot as plt
import numpy as np

def plot_trajectory_with_speed(pose_data, body_part="centroid"):
    """Plot a trajectory with speed indicated by color.

    Parameters
    ----------
    pose_data : movement.Pose
        The pose dataset containing position data.
    body_part : str, optional
        The body part to plot (default: "centroid").
    """
    # Extract x, y coordinates from Pose object
    data = pose_data.data[body_part].values  # Adjust based on actual Pose structure
    x, y = data[:, 0], data[:, 1]  # Assuming [n_frames, 2] shape
    # Calculate speed (Euclidean distance between consecutive points)
    speeds = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    # Plot trajectory line
    plt.plot(x, y, "o-", label="Trajectory", alpha=0.5)
    # Overlay speed as color
    plt.scatter(x[:-1], y[:-1], c=speeds, cmap="viridis", label="Speed")
    plt.colorbar(label="Speed (units/frame)")
    plt.title(f"{body_part.capitalize()} Trajectory with Speed")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()
    plt.show()