import numpy as np


def load_keypoints(file_path, fps=None, timestamps=None):
    """Load keypoints from the given file and support time coordinates.

    Parameters
    ----------
    - file_path (str): Path to the keypoints file.
    - fps (float, optional): Frames per second, used if timestamps are not provided.
    - timestamps (list or np.array, optional): Explicit timestamps for each frame.

    Returns
    -------
    - keypoints (np.array): Loaded keypoints.
    - times (np.array): Corresponding time coordinates.

    """
    # Load keypoints (assuming CSV format for now)
    keypoints = np.loadtxt(file_path, delimiter=",")

    if timestamps is not None:
        times = np.array(timestamps)
        if len(times) != len(keypoints):
            raise ValueError(
                "Length of timestamps must match the number of keypoint frames."
            )
    elif fps is not None:
        times = np.arange(len(keypoints)) / fps
    else:
        raise ValueError("Either 'fps' or 'timestamps' must be provided.")

    return keypoints, times
