"""
Converter: Pose dataset → Bounding boxes dataset
"""

import numpy as np
import xarray as xr


def poses_to_bboxes(
    poses_ds: xr.Dataset,
    padding: float = 0.0,
) -> xr.Dataset:
    """
    Convert pose keypoints dataset into bounding boxes dataset.

    Parameters
    ----------
    poses_ds : xr.Dataset
        Pose dataset with dims:
        (time, individuals, keypoints, coords)

    padding : float
        Optional padding added to bbox edges.

    Returns
    -------
    xr.Dataset
        Bounding boxes dataset.
    """

    # Extract coordinates
    coords = poses_ds["position"]  # adjust if different name

    time_dim = coords.sizes["time"]
    indiv_dim = coords.sizes["individuals"]

    bboxes = []

    for t in range(time_dim):
        frame_boxes = []

        for i in range(indiv_dim):
            keypoints = coords.isel(time=t, individuals=i).values

            # Remove NaNs
            keypoints = keypoints[~np.isnan(keypoints).any(axis=1)]

            if len(keypoints) == 0:
                frame_boxes.append([np.nan] * 4)
                continue

            x_min = keypoints[:, 0].min() - padding
            y_min = keypoints[:, 1].min() - padding
            x_max = keypoints[:, 0].max() + padding
            y_max = keypoints[:, 1].max() + padding

            frame_boxes.append([x_min, y_min, x_max, y_max])

        bboxes.append(frame_boxes)

    bboxes = np.array(bboxes)

    bbox_ds = xr.Dataset(
        {
            "bboxes": (
                ["time", "individuals", "bbox_coords"],
                bboxes,
            )
        },
        coords={
            "bbox_coords": ["x_min", "y_min", "x_max", "y_max"]
        },
    )

    return bbox_ds