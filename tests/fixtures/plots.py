import numpy as np
import pytest
import xarray as xr


@pytest.fixture
def one_individual():
    """Sample data for plot testing.

    Data has five keypoints for one cross shaped mouse that is centered
    around the origin and moves forwards along the positive y axis with
    steps of 1.

    Keypoint starting position (x, y):
    - left (-1, 0)
    - centre (0, 0)
    - right (1, 0)
    - snout (0, 1)
    - tail (0, -1)

    """
    time_steps = 4
    individuals = ["id_0"]
    keypoints = ["left", "centre", "right", "snout", "tail"]
    space = ["x", "y"]
    positions = {
        "left": {"x": -1, "y": np.arange(time_steps)},
        "centre": {"x": 0, "y": np.arange(time_steps)},
        "right": {"x": 1, "y": np.arange(time_steps)},
        "snout": {"x": 0, "y": np.arange(time_steps) + 1},
        "tail": {"x": 0, "y": np.arange(time_steps) - 1},
    }

    time = np.arange(time_steps)
    position_data = np.zeros(
        (time_steps, len(space), len(keypoints), len(individuals))
    )

    # Create x and y coordinates arrays
    x_coords = np.array([positions[key]["x"] for key in keypoints])
    y_coords = np.array([positions[key]["y"] for key in keypoints])

    for i, _ in enumerate(keypoints):
        position_data[:, 0, i, 0] = x_coords[i]  # x-coordinates
        position_data[:, 1, i, 0] = y_coords[i]  # y-coordinates

    da = xr.DataArray(
        position_data,
        name="position",
        dims=["time", "space", "keypoint", "individual"],
        coords={
            "time": time,
            "space": space,
            "keypoint": keypoints,
            "individual": individuals,
        },
    )
    return da


@pytest.fixture
def two_individuals(one_individual):
    """Return a position array with two cross-shaped mice.

    The 0-th mouse is moving forwards along the positive y axis, i.e. same as
    in sample_data_one_cross, the 1-st mouse is moving in the opposite
    direction, i.e. with it's snout towards the negative side of the y axis.

    The left and right keypoints are not mirrored for id_1, so this
    mouse is moving flipped around on it's back.
    """
    da_id1 = one_individual.copy()
    da_id1.loc[dict(space="y")] = da_id1.sel(space="y") * -1
    da_id1 = da_id1.assign_coords(individual=["id_1"])
    return xr.concat([one_individual.copy(), da_id1], "individual")
