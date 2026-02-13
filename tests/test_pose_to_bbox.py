import numpy as np
import xarray as xr

from movement.io import poses_to_bboxes


def test_conversion():
    data = np.array(
        [
            [[[10, 10], [20, 20]]],  # frame 1
            [[[15, 15], [25, 25]]],  # frame 2
        ]
    )

    ds = xr.Dataset(
        {
            "position": (
                ["time", "individuals", "keypoints", "coords"],
                data,
            )
        }
    )

    bbox_ds = poses_to_bboxes(ds)

    assert bbox_ds is not None
